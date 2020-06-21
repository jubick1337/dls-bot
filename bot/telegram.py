import logging
import ssl
from pathlib import Path

import telebot
from aiohttp import web

TELEGRAM_TOKEN = '1109017372:AAG7i7w4mDWlY9oUqaDQW8rN2NM105OODZ4'

WEBHOOK_HOST = '35.204.74.60'
WEBHOOK_PORT = 8443
WEBHOOK_LISTEN = '0.0.0.0'

WEBHOOK_SSL_CERT = Path('../key/url_cert.pem')  # Path to the ssl certificate
WEBHOOK_SSL_PRIV = Path('../key/url_private.key')  # Path to the ssl private key

WEBHOOK_URL_BASE = f'https://{WEBHOOK_HOST}:{WEBHOOK_PORT}'
WEBHOOK_URL_PATH = f'/{TELEGRAM_TOKEN}/'

logger = telebot.logger
telebot.logger.setLevel(logging.INFO)

bot = telebot.TeleBot(TELEGRAM_TOKEN)

app = web.Application()


async def handle(request):
    if request.match_info.get('token') == bot.token:
        request_body_dict = await request.json()
        update = telebot.types.Update.de_json(request_body_dict)
        bot.process_new_updates([update])
        return web.Response()
    else:
        return web.Response(status=403)


app.router.add_post('/{token}/', handle)


@bot.message_handler(commands=['start'])
def greet(message):
    bot.reply_to(message, "Hi there, type one of commands: /nst")


@bot.message_handler(commands=['nst'])
def start_nst(message):
    bot.reply_to(message, 'Now send me two photos. First for content and second for style.')


bot.remove_webhook()
bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH,
                certificate=open(WEBHOOK_SSL_CERT, 'r'))

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain(WEBHOOK_SSL_CERT, WEBHOOK_SSL_PRIV)

web.run_app(
    app,
    host=WEBHOOK_LISTEN,
    port=WEBHOOK_PORT,
    ssl_context=context,
)
