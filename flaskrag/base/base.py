import re
import traceback

import plotly
import plotly.express as px
import plotly.graph_objects as go
import sqlparse
from pandas import DataFrame

from .util import DatabaseClient, EmbeddingGenerator, LLMClient, VectorDBClient


class BaseRAG:
    def __init__(
        self,
        vdb: VectorDBClient,
        embedding_generator: EmbeddingGenerator,
        llm: LLMClient,
        db: DatabaseClient,
        config: dict = None,
    ):
        self.vdb = vdb
        self.embedding_generator = embedding_generator
        self.llm = llm
        self.db = db
        self.config = config if config else {}

    def log(self, message: str, title: str = "Info"):
        print(f"{title}: {message}")

    def generate_sql(self, question: str, **kwargs) -> str:
        """
        Generates an SQL query based on a given question and additional context.
        This method uses a language model to generate an SQL query by first creating a prompt
        with relevant information such as similar questions, related DDL statements, and
        documentation. It then submits this prompt to the language model to get an initial
        response. If the response contains intermediate SQL, it runs this SQL and uses the
        results to generate a final SQL query.
        Args:
            question (str): The question for which to generate the SQL query.
            **kwargs: Additional keyword arguments to pass to various methods.
        Returns:
            str: The generated SQL query.
        Raises:
            Exception: If there is an error running the intermediate SQL query.
        """
        opts = {
            "initial_prompt": self.config.get("initial_prompt", None),
            "question_sql_list": self.vdb.get_similar_question(question, **kwargs),
            "ddl_list": self.vdb.get_related_ddl(question, **kwargs),
            "doc_list": self.vdb.get_related_documentation(question, **kwargs),
            "question": question,
        }
        prompt = self.llm.get_sql_prompt(**opts)
        self.log(title="SQL Prompt", message=prompt)
        llm_response = self.llm.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        if "intermediate_sql" in llm_response:
            intermediate_sql = self.extract_sql(llm_response)

            try:
                self.log(title="Running Intermediate SQL", message=intermediate_sql)
                df = self.run_sql(intermediate_sql)

                opts["doc_list"].append(
                    f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n{df.to_markdown()}"
                )
                prompt = self.llm.get_sql_prompt(**opts)
                self.log(title="Final SQL Prompt", message=prompt)

                llm_response = self.llm.submit_prompt(prompt, **kwargs)
                self.log(title="LLM Response", message=llm_response)
            except Exception as e:
                return f"Error running intermediate SQL: {e}"

        return self.extract_sql(llm_response)

    def extract_sql(self, llm_response: str) -> str:
        """
        Extracts the last SQL query from a given LLM (Language Model) response string.

        The method searches for SQL queries within the provided response using multiple patterns:
        1. Markdown code blocks (with or without the `sql` tag).
        2. Plain SQL queries starting with `SELECT` and ending with a semicolon.
        3. SQL queries containing Common Table Expressions (CTEs) starting with `WITH` and ending with a semicolon.

        If multiple SQL queries are found, the last one is returned. If no SQL query is found, the original response is returned.

        Args:
            llm_response (str): The response string from the LLM containing potential SQL queries.

        Returns:
            str: The extracted SQL query or the original response if no SQL query is found.
        """
        sqls = (
            re.findall(r"```(.*)```", llm_response, re.DOTALL)
            # If the llm_response contains a markdown code block, with or without the sql tag, extract the last sql from it
            + re.findall(r"```sql\n(.*)```", llm_response, re.DOTALL)
            # If the llm_response is not markdown formatted, extract last sql by finding select and ; in the response
            + re.findall(r"SELECT.*?;", llm_response, re.DOTALL)
            # If the llm_response contains a CTE (with clause), extract the last sql between WITH and ;
            + re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL)
        )
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql
        return llm_response

    def is_valid_sql(self, sql: str) -> bool:
        for statement in sqlparse.parse(sql):
            if statement.get_type() == "SELECT":
                return True

        return False

    def run_sql(self, sql: str) -> DataFrame:
        return self.db.run_sql(sql)

    def generate_summary(self, question: str, df: DataFrame, **kwargs) -> str:
        return self.llm.generate_summary(question=question, df=df, stream=True, **kwargs)

    def should_generate_chart(self, df: DataFrame) -> bool:
        if not df.empty and df.select_dtypes(include=["number"]).shape[1] > 0:
            return True

        return False

    def ask(
        self,
        question: str | None = None,
        print_results: bool = True,
        visualise: bool = True,
        allow_llm_to_see_data: bool = False,
    ) -> tuple[str | None, DataFrame | None, plotly.graph_objs.Figure | None]:
        """
        Asks a question, generates an SQL query, executes it, and optionally visualizes the results.

        This method takes a question as input, generates an SQL query based on the question, executes the query,
        and returns the SQL query, the results of the query as a DataFrame, and an optional Plotly figure for visualization.

            question (str | None): The question to ask. If None, prompts the user for input.
            print_results (bool): Whether to print the results of the SQL query. Defaults to True.
            visualise (bool): Whether to generate Plotly code and display the Plotly figure. Defaults to True.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data for generating SQL. Defaults to False.

            tuple[str, DataFrame, Figure]:
                - The SQL query as a string, or None if an error occurred.
                - The results of the SQL query as a DataFrame, or None if an error occurred.
                - The Plotly figure for visualization, or None if visualization is disabled or an error occurred.
        Args:
            question (str): The question to ask.
            print_results (bool): Whether to print the results of the SQL query.
            visualise (bool): Whether to generate plotly code and display the plotly figure.

        Returns:
            Tuple[str, DataFrame, Figure]: The SQL query, the results of the SQL query, and the plotly figure.
        """

        question = question or input("Enter a question: ")

        try:
            sql = self.generate_sql(question=question, allow_llm_to_see_data=allow_llm_to_see_data)
        except Exception as e:
            print(e)
            return None, None, None

        if print_results:
            print(sql)

        try:
            df = self.db.run_sql(sql)

            if print_results:
                print(df)

            # generate plotly code if visualise is True
            if visualise:
                try:
                    plotly_code = self.llm.generate_plotly_code(
                        question=question,
                        sql=sql,
                        df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                    )
                    fig = self.get_plotly_figure(plotly_code=plotly_code, df=df)
                    if print_results:
                        fig.show()
                except Exception as e:
                    # Print stack trace
                    traceback.print_exc()
                    print("Couldn't run plotly code: ", e)
                    return sql, df, None
            else:
                return sql, df, None

        except Exception as e:
            print("Couldn't run sql: ", e)
            return sql, None, None

        return sql, df, fig

    def train(
        self,
        questions: dict[str, str] = None,
        ddls: list[str] = None,
        documentation: list[str] = None,
    ):
        """
        Trains the model using provided questions, DDL statements, and documentation.

        This method allows you to train the model by adding SQL question pairs, DDL statements,
        and documentation. Each type of input is optional and will be processed if provided.

            questions (dict[str, str], optional): The question-SQL pairs to train on.
            ddls (list[str], optional): The DDL statements to train on.
            documentation (list[str], optional): The documentation to train on.
        Args:
            question (dict[str, str]): The question-sql pairs to train on.
            ddls (list(str)):  The DDL statement.
            documentation (list(str)): The documentation to train on.
        """
        # todo: auto-train using, say, infomation_schema from the database
        if documentation:
            print("Adding documentation...")
            [self.vdb.add_documentation(doc) for doc in documentation]

        if questions:
            print("Adding SQL...")
            [self.add_question_sql(**question) for question in questions]

        if ddls:
            print("Adding ddl...")
            [self.add_ddl(ddl) for ddl in ddls]

    def generate_plotly_code(self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs) -> str:
        return self.llm.generate_plotly_code(question=question, sql=sql, df_metadata=df_metadata, **kwargs)

    def get_plotly_figure(self, plotly_code: str, df: DataFrame, dark_mode: bool = True) -> plotly.graph_objs.Figure:
        """
        Get a Plotly figure from a dataframe and Plotly code.

        Args:
            df (DataFrame): The dataframe to use.
            plotly_code (str): The Plotly code to use.

        Returns:
            plotly.graph_objs.Figure: The Plotly figure.
        """
        ldict = {"df": df, "px": px, "go": go}
        try:
            exec(plotly_code, globals(), ldict)

            fig = ldict.get("fig", None)
        except Exception as e:
            print("Couldn't run plotly code: ", e)
            print("Falling back to deterministic plot type selection...")
            # Inspect data types
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            # Decision-making for plot type
            if len(numeric_cols) >= 2:
                # Use the first two numeric columns for a scatter plot
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
                # Use a bar plot if there's one numeric and one categorical column
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
            elif len(categorical_cols) >= 1 and df[categorical_cols[0]].nunique() < 10:
                # Use a pie chart for categorical data with fewer unique values
                fig = px.pie(df, names=categorical_cols[0])
            else:
                # Default to a simple line plot if above conditions are not met
                fig = px.line(df)

        if fig and dark_mode:
            fig.update_layout(template="plotly_dark")

        return fig
