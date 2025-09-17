import re
from abc import ABC, abstractmethod

from pandas import DataFrame, read_sql_query
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


class DatabaseClient:
    def __init__(self, url: str, init_sql: str = None, **kwargs):
        """
        Args:
            url (str): The URL of the database to connect to.
            init_sql (str, optional): SQL to run when connecting to the database. Defaults to None.
        """
        self.url = url
        self.init_sql = init_sql
        self.kwargs = kwargs
        self.engine = None
        self.session = None
        self.dialect = None
        self._connect()

    def _connect(self):
        self.engine = create_engine(self.url, **self.kwargs)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        self.dialect = self.engine.dialect.name

        if self.init_sql:
            self.session.execute(self.init_sql)
            self.session.commit()

    def run_sql(self, sql: str, **kwargs) -> DataFrame:
        """
        Run a SQL query and return the result as a DataFrame.

        Args:
            sql (str): The SQL query to run.

        Returns:
            DataFrame: The result of the query.
        """
        with self.engine.begin() as conn:
            df = read_sql_query(text(sql), conn, **kwargs)
            conn.close()
            return df


class VectorDBClient(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_similar_question(self, question: str, **kwargs) -> list:
        """
        This method is used to get similar questions and their corresponding SQL statements.

        Args:
            question (str): The question to get similar questions and their corresponding SQL statements for.

        Returns:
            list: A list of similar questions and their corresponding SQL statements.
        """
        pass

    @abstractmethod
    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        This method is used to get related DDL statements to a question.

        Args:
            question (str): The question to get related DDL statements for.

        Returns:
            list: A list of related DDL statements.
        """
        pass

    @abstractmethod
    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        This method is used to get related documentation to a question.

        Args:
            question (str): The question to get related documentation for.

        Returns:
            list: A list of related documentation.
        """
        pass

    @abstractmethod
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
        This method is used to add a question and its corresponding SQL query to the training data.

        Args:
            question (str): The question to add.
            sql (str): The SQL query to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_ddl(self, ddl: str, **kwargs) -> str:
        """
        This method is used to add a DDL statement to the training data.

        Args:
            ddl (str): The DDL statement to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        This method is used to add documentation to the training data.

        Args:
            documentation (str): The documentation to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def get_training_data(self, **kwargs) -> DataFrame:
        """
        This method is used to get all the training data from the retrieval layer.

        Returns:
            DataFrame: The training data.
        """
        pass

    @abstractmethod
    def remove_training_data(self, id: str, **kwargs) -> bool:
        """
        This method is used to remove training data from the retrieval layer.

        Args:
            id (str): The ID of the training data to remove.

        Returns:
            bool: True if the training data was removed, False otherwise.
        """
        pass


class EmbeddingGenerator(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def generate_embedding(self, data: str) -> list:
        # Code to generate embedding using the model
        pass


class LLMClient(ABC):
    def __init__(self, api_key, **kwargs):
        self.api_key = api_key
        self.max_tokens = kwargs.get("max_tokens", 14000)
        self.static_documentation = kwargs.get("static_documentation", None)
        self.language = kwargs.get("language", "English")

    @abstractmethod
    def system_message(self, message: str) -> any:
        pass

    @abstractmethod
    def user_message(self, message: str) -> any:
        pass

    @abstractmethod
    def assistant_message(self, message: str) -> any:
        pass

    def str_to_approx_token_count(self, string: str) -> int:
        return len(string) / 4

    def _response_language(self) -> str:
        return f"Respond in the {self.language} language." if self.language else ""

    def _add_to_prompt(self, initial_prompt: str, list_to_add: list[str]) -> str:
        for item in list_to_add:
            if self.str_to_approx_token_count(initial_prompt) + self.str_to_approx_token_count(item) < self.max_tokens:
                initial_prompt += f"{item}\n\n"
            else:
                break
        return initial_prompt

    def add_ddl_to_prompt(self, initial_prompt: str, ddl_list: list[str]) -> str:
        if ddl_list:
            initial_prompt += "\n===Tables \n"
            initial_prompt += self._add_to_prompt(initial_prompt, ddl_list)

        return initial_prompt

    def add_documentation_to_prompt(self, initial_prompt: str, documentation_list: list[str]) -> str:
        if documentation_list:
            initial_prompt += "\n===Additional Context \n\n"
            initial_prompt += self._add_to_prompt(initial_prompt, documentation_list)

        return initial_prompt

    def add_sql_to_prompt(self, initial_prompt: str, sql_list: dict[str]) -> str:
        if sql_list:
            initial_prompt += "\n===Question-SQL Pairs\n\n"
            initial_prompt += self._add_to_prompt(
                initial_prompt, [f"{question['question']}\n{question['sql']}" for question in sql_list]
            )

        return initial_prompt

    def get_sql_prompt(
        self,
        initial_prompt: str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        dialect: str = "SQL",
    ):
        if initial_prompt is None:
            initial_prompt = (
                f"You are a {dialect} expert. "
                + "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "
            )

        initial_prompt = self.add_ddl_to_prompt(initial_prompt, ddl_list)

        if self.static_documentation:
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(initial_prompt, doc_list)

        initial_prompt += (
            "===Response Guidelines \n"
            "1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n"
            "2. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
            "3. If the provided context is insufficient, please explain why it can't be generated. \n"
            "4. Please use the most relevant table(s). \n"
            "5. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
            f"6. Ensure that the output SQL is {dialect}-compliant and executable, and free of syntax errors. \n"
        )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log

    def get_followup_questions_prompt(
        self,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
    ) -> list:
        initial_prompt = f"The user initially asked the question: '{question}': \n\n"
        initial_prompt += self.add_ddl_to_prompt(initial_prompt, ddl_list)
        initial_prompt += self.add_documentation_to_prompt(initial_prompt, doc_list)
        initial_prompt += self.add_sql_to_prompt(initial_prompt, question_sql_list)

        message_log = [self.system_message(initial_prompt)]
        message_log.append(
            self.user_message(
                "Generate a list of followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions."
            )
        )

        return message_log

    @abstractmethod
    def submit_prompt(self, prompt, **kwargs) -> str:
        """
        This method is used to submit a prompt to the LLM.

        Args:
            prompt (any): The prompt to submit to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

    def generate_question(self, sql: str, **kwargs) -> str:
        response = self.submit_prompt(
            [
                self.system_message(
                    "The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."
                ),
                self.user_message(sql),
            ],
            **kwargs,
        )

        return response

    def generate_summary(self, question: str, df: DataFrame, **kwargs) -> str:
        """
        Generate a summary of the results of a SQL query.

        Args:
            question (str): The question that was asked.
            df (DataFrame): The results of the SQL query.

        Returns:
            str: The summary of the results of the SQL query.
        """

        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe following is a pandas DataFrame with the results of the query: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                "Briefly summarize the data based on the question that was asked. Do not respond with any additional explanation beyond the summary."
                + self._response_language()
            ),
        ]

        summary = self.submit_prompt(message_log, **kwargs)

        return summary

    def combine_questions(self, last_question: str, new_question: str, **kwargs) -> str:
        if not last_question:
            return new_question

        prompt = [
            self.llm.system_message(
                "Your goal is to combine a sequence of questions into a singular question if they are related. If the second question does not relate to the first question and is fully self-contained, return the second question. Return just the new combined question with no additional explanations. The question should theoretically be answerable with a single SQL statement."
            ),
            self.llm.user_message("First question: " + last_question + "\nSecond question: " + new_question),
        ]

        return self.llm.submit_prompt(prompt=prompt, **kwargs)

    def generate_followup_questions(
        self, question: str, sql: str, df: DataFrame, n_questions: int = 5, **kwargs
    ) -> list:
        message_log = [
            self.llm.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe SQL query for this question was: {sql}\n\nThe following is a pandas DataFrame with the results of the query: \n{df.to_markdown()}\n\n"
            ),
            self.llm.user_message(
                f"Generate a list of {n_questions} followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions. Remember that there should be an unambiguous SQL query that can be generated from the question. Prefer questions that are answerable outside of the context of this conversation. Prefer questions that are slight modifications of the SQL query that was generated that allow digging deeper into the data. Each question will be turned into a button that the user can click to generate a new SQL query so don't use 'example' type questions. Each question must have a one-to-one correspondence with an instantiated SQL query."
                + self._response_language()
            ),
        ]

        llm_response = self.submit_prompt(message_log, **kwargs)

        serial_nos_removed = re.sub(r"^\d+\.\s*", "", llm_response, flags=re.MULTILINE)
        return serial_nos_removed.split("\n")

    def _extract_python_code(self, markdown_string: str) -> str:
        # Regex pattern to match Python code blocks
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

        # Find all matches in the markdown string
        matches = re.findall(pattern, markdown_string, re.IGNORECASE)

        # Extract the Python code from the matches
        python_code = []
        for match in matches:
            python = match[0] if match[0] else match[1]
            python_code.append(python.strip())

        if python_code:
            return python_code[0]

        return markdown_string

    def _sanitize_plotly_code(self, raw_plotly_code: str) -> str:
        # Remove the fig.show() statement from the plotly code
        plotly_code = raw_plotly_code.replace("fig.show()", "")

        return plotly_code

    def generate_plotly_code(self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs) -> str:
        if question is not None:
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "Can you generate the Python plotly code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python code. Do not answer with any explanations -- just the code."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))
