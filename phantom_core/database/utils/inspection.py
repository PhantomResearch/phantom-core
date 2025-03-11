from sqlalchemy import Engine, inspect

def listtables(engine: Engine) -> list[str]:
    insp = inspect(engine)
    return insp.get_table_names()