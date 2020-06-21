import logging
import ssl

import telebot
from aiohttp import web
from telebot.types import Message

from bot import db_worker
from bot.utils import WEBHOOK_URL_BASE, WEBHOOK_URL_PATH, WEBHOOK_SSL_CERT, WEBHOOK_SSL_PRIV, TELEGRAM_TOKEN, States, \
    WEBHOOK_LISTEN, WEBHOOK_PORT
from model.nst import NST

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
def greet(message: Message):
    bot.reply_to(message, 'Hi there, type one of commands: /nst')
    db_worker.set_state(message.chat.id, States.START.value)


@bot.message_handler(commands=['reset'])
def cmd_reset(message: Message):
    bot.send_message(message.chat.id, 'Hi there, type one of commands: /nst')
    db_worker.set_state(message.chat.id, States.START.value)


@bot.message_handler(commands=['nst'])
def start_nst(message: Message):
    bot.reply_to(message, 'Now send me two photos. First for content and second for style.')
    db_worker.set_state(message.chat.id, States.ENTER_FIRST_PIC.value)


@bot.message_handler(func=lambda message: db_worker.get_current_state(message.chat.id) == States.ENTER_FIRST_PIC.value)
def get_content(message: Message):
    if len(message.photo == 0):
        bot.reply_to(message, 'Something went wrong. Try again:(')
        return

    downloaded_file = bot.download_file(bot.get_file(message.photo[-1].file_id))

    with open(f'content{message.chat.id}', 'wb') as file:
        file.write(downloaded_file)

    bot.send_message(message.chat.id, 'now send me second photo')
    db_worker.set_state(message.chat.id, States.ENTER_SECOND_PIC.value)


@bot.message_handler(func=lambda message: db_worker.get_current_state(message.chat.id) == States.ENTER_SECOND_PIC.value)
def get_content(message: Message):
    if len(message.photo == 0):
        bot.reply_to(message, 'Something went wrong. Try again:(')
        return

    downloaded_file = bot.download_file(bot.get_file(message.photo[-1].file_id))

    with open(f'style{message.chat.id}', 'wb') as file:
        file.write(downloaded_file)

    model = NST(128)
    res = await model.transform(f'content{message.chat.id}.jpg', f'style{message.chat.id}.jpg')
    model.unload(res).save(f'res{message.chat.id}.jpg')

    bot.send_photo(message.chat.id, f'res{message.chat.id}.jpg')
    db_worker.set_state(message.chat.id, States.START.value)


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
