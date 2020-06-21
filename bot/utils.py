from enum import Enum
from pathlib import Path

db_file = 'database.vdb'

TELEGRAM_TOKEN = '1109017372:AAG7i7w4mDWlY9oUqaDQW8rN2NM105OODZ4'

WEBHOOK_HOST = '35.204.74.60'
WEBHOOK_PORT = 8443
WEBHOOK_LISTEN = '0.0.0.0'

WEBHOOK_SSL_CERT = Path('../key/url_cert.pem')
WEBHOOK_SSL_PRIV = Path('../key/url_private.key')

WEBHOOK_URL_BASE = f'https://{WEBHOOK_HOST}:{WEBHOOK_PORT}'
WEBHOOK_URL_PATH = f'/{TELEGRAM_TOKEN}/'


class States(Enum):
    START = '0'
    ENTER_COMMAND = '1'
    ENTER_FIRST_PIC = '2'
    ENTER_SECOND_PIC = '3'
    FINISH = '4'
