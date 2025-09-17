import json
from functools import wraps

import flask
from flask import Flask, Response, jsonify, request
from flask_sock import Sock

from ..base import BaseRAG
from .cache import Cache, MemoryCache

VERSION = "v0"


class APIServer:
    """
    Template API server for the RAG pipeline.
    """

    def requires_cache(self, required_fields, optional_fields=[]):
        """
        Decorator to ensure that certain fields are present in the cache before proceeding with the decorated function.

        TODO:
            return the cached value if it exists..?

        Args:
            required_fields (list): List of fields that must be present in the cache.
            optional_fields (list, optional): List of fields that are optional. Defaults to an empty list.

        Returns:
            function: The decorated function with cache validation.
        """

        def decorator(f):
            """
            Inner decorator function that wraps the original function.

            Args:
                f (function): The function to be decorated.

            Returns:
                function: The wrapped function with cache validation.
            """

            @wraps(f)
            def decorated(*args, **kwargs):
                """
                Wrapper function that performs the cache validation.

                Args:
                    *args: Positional arguments for the wrapped function.
                    **kwargs: Keyword arguments for the wrapped function.

                Returns:
                    Response: JSON response indicating error if validation fails, otherwise the result of the wrapped function.
                """
                id = request.args.get("id")

                if id is None:
                    id = request.json.get("id")
                    if id is None:
                        return jsonify({"type": "error", "error": "No id provided"})

                for field in required_fields:
                    if self.cache.get(id=id, field=field) is None:
                        return jsonify({"type": "error", "error": f"No {field} found"})

                field_values = {field: self.cache.get(id=id, field=field) for field in required_fields}

                for field in optional_fields:
                    field_values[field] = self.cache.get(id=id, field=field)

                # Add the id to the field_values
                field_values["id"] = id

                return f(*args, **field_values, **kwargs)

            return decorated

        return decorator

    def __init__(self, rag: BaseRAG, debug: bool = False, cache: Cache = MemoryCache()):
        self.app = Flask(__name__)
        self.sock = Sock(self.app)
        self.ws_clients = []
        self.debug = debug
        self.rag = rag
        self.cache = cache

        if self.debug:

            def log(message, title="Info"):
                [ws.send(json.dumps({"message": message, "title": title})) for ws in self.ws_clients]

            self.rag.log = log

        @self.app.get(f"/api/{VERSION}/generate_sql")
        def generate_sql():
            question = flask.request.args.get("question")

            if not question:
                return jsonify({"type": "error", "error": "No question provided"})

            id = self.cache.generate_id(question=question)
            sql = rag.generate_sql(question=question)

            self.cache.set(id=id, field="question", value=question)
            self.cache.set(id=id, field="sql", value=sql)

            if rag.is_valid_sql(sql=sql):
                return jsonify(
                    {
                        "type": "sql",
                        "id": id,
                        "text": sql,
                    }
                )

            return {
                "type": "text",
                "id": id,
                "text": sql,
            }

        @self.app.get(f"/api/{VERSION}/run_sql")
        @self.requires_cache(["sql"])
        def run_sql(id: str, sql):
            try:
                sql = flask.request.args.get("sql")

                if not sql:
                    return jsonify({"type": "error", "error": "No SQL provided"})

                df = rag.run_sql(sql)
                return jsonify(
                    {
                        "type": "dataframe",
                        "id": id,
                        "df": df.to_dict(),
                        "should_generate_chart": rag.should_generate_chart(df),
                    }
                )
            except Exception as e:
                return jsonify({"type": "error", "error": str(e)})

        @self.app.get(f"/api/{VERSION}/download_csv")
        @self.requires_cache(["df"])
        def download_csv(user: any, id: str, df):
            csv = df.to_csv()

            return Response(
                csv,
                mimetype="text/csv",
                headers={"Content-disposition": f"attachment; filename={id}.csv"},
            )

        @self.app.get(f"/api/{VERSION}/generate_plotly_figure")
        @self.requires_cache(["df", "question", "sql"])
        def generate_plotly_figure(id: str, df, question, sql):
            chart_instructions = flask.request.args.get("chart_instructions")

            try:
                # If chart_instructions is not set then attempt to retrieve the code from the cache
                if not chart_instructions:
                    code = self.cache.get(id=id, field="plotly_code")
                else:
                    question = (
                        f"{question}. When generating the chart, use these special instructions: {chart_instructions}"
                    )
                    code = rag.generate_plotly_code(
                        question=question,
                        sql=sql,
                        df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                    )
                    self.cache.set(id=id, field="plotly_code", value=code)

                fig = rag.get_plotly_figure(plotly_code=code, df=df, dark_mode=True)
                fig_json = fig.to_json()

                self.cache.set(id=id, field="fig_json", value=fig_json)

                return jsonify(
                    {
                        "type": "plotly_figure",
                        "id": id,
                        "fig": fig_json,
                    }
                )
            except Exception as e:
                # Print the stack trace
                import traceback

                traceback.print_exc()

                return jsonify({"type": "error", "error": str(e)})

        @self.app.get(f"/api/{VERSION}/generate_summary")
        @self.requires_cache(["df", "question"])
        def generate_summary(user: any, id: str, df, question):
            summary = rag.generate_summary(question=question, df=df)

            def streaming():
                resp = ""
                for chunk in summary:
                    chunk = chunk.replace("\n", "<br>")
                    resp += chunk
                    yield f"data: {jsonify({"type": "text", "id": id, "text": resp})}\n\n"
                else:
                    self.cache.set(id=id, field="summary", value=resp)

            return streaming()

    def run(self):
        self.app.run()
