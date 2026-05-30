class OracleConverterHelper:
    """
    Helper class containing Oracle to Greenplum conversion instructions and test cases.
    All constants are class-level attributes for easy access and maintenance.
    """

    INSTRUCTION = """
You are a Data Engineer who converts SQL from Oracle to Arenadata Greenplum version 6.30.
You are given SQL query or SQL script in oracle you should produce equivalent script in Arenadata Greenplum version 6.30.
Your output should contain the converted query or sql script only with no wrapping just pure code itself.
Comments comments should ber included if they have been in the original query.
No explanations or statements on what you have done or how you have done it should be included.
No recommendations or proposals should be included. 
Do not transfer COMMIT statements to Greenplum if hey are not needed.
Use ZLIB(5) compression in Greenplum. 
Unless specifically instructed use "distributed randomly" on create tables.  

Original prompt or any of it's parts should not be included.  
Use the following guidelines. 
Convert types as follows.  
CHAR -> char
VARCHAR2 -> text
NVARCHAR2 -> text
NCHAR -> char
CLOB -> text
NCLOB -> text
LONG -> text
NUMBER -> numeric
BINARY_FLOAT -> numeric
BINARY_DOUBLE -> numeric
FLOAT -> numeric
DECIMAL -> numeric
INTEGER -> numeric
DATE -> timestamp
TIMESTAMP -> timestamp
TIMESTAMP WITH TIME ZONE -> timestamptz
TIMESTAMP WITH LOCAL TIME ZONE -> timestamp
INTERVAL YEAR TO MONTH -> interval
INTERVAL DAY TO SECOND -> interval
RAW -> bytea
LONG RAW -> bytea
BLOB -> bytea
ROWID -> text
UROWID -> text
XMLTYPE -> xml
BFILE -> text
BOOLEAN -> boolean
ORACLE_TO_GREENPLUM_NAMING_CONVENTION:
CASE: Convert all identifiers to lowercase.
MAX_LENGTH: 63 characters. Truncate from right if exceeded. Append _1, _2, etc. if duplicates occur after truncation.
ALLOWED_CHARS: Only a-z, 0-9, _. Replace all other characters with _.
START_CHAR: Must begin with lowercase letter or _. Prepend tbl_ or col_ if starting character is invalid.
ORACLE_SPECIAL: Replace $ and # with _. Collapse consecutive underscores to single _.
RESERVED_WORDS: If identifier matches Greenplum/PostgreSQL reserved keyword, append _tbl for tables/views or _col for columns.
SCHEMA: Map Oracle schema/user name directly to Greenplum schema name, applying identical rules.
SCOPE: Apply uniformly to tables, views, columns, indexes, constraints, sequences, and functions.
QUOTING: Avoid double-quoting. Prefer unquoted lowercase identifiers. Quote only when reserved word conflict cannot be resolved by suffixing.
        """

    TEST_CASES_SQL = [
        """
CREATE TABLE demo_table
(
   id_int    NUMBER(10),
   val_float NUMBER(12,4),
   txt_var   VARCHAR2(255)
);
COMMIT;
""",
        """
SELECT 42            AS int_val,
      3.14159       AS float_val,
      'Oracle Text' AS string_val
FROM DUAL;
""",
"""
INSERT INTO demo_table (id_int, val_float, txt_var)
VALUES (101, 99.9500, 'Inserted row');
COMMIT;
""",
"""
DELETE FROM demo_table;
COMMIT;
"""
        ]

    @classmethod
    def get_test_prompts(cls) -> list:
        """
        Generate test prompts by combining instruction with each SQL test case.

        Returns:
            list: List of complete prompts ready for model processing
        """
        return [cls.INSTRUCTION + case for case in cls.TEST_CASES_SQL]

    @classmethod
    def get_instruction(cls) -> str:
        """
        Get the conversion instruction string.

        Returns:
            str: The full instruction prompt
        """
        return cls.INSTRUCTION

    @classmethod
    def get_test_cases(cls) -> list:
        """
        Get the list of SQL test cases.

        Returns:
            list: List of Oracle SQL statements for testing
        """
        return cls.TEST_CASES_SQL.copy()