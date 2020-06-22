from vedis import Vedis

from tg.utils import States, db_file


def get_current_state(user_id: str) -> States:
    with Vedis(db_file) as db:
        try:
            return db[user_id].decode()
        except KeyError:
            return States.START.value


def set_state(user_id: str, value: str) -> bool:
    with Vedis(db_file) as db:
        try:
            db[user_id] = value
            return True
        except:
            return False
