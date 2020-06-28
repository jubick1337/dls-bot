import logging
import os
import ssl
import time

import telebot
from aiohttp import web
from aiohttp.abc import Request
from telebot import apihelper
from telebot.types import Message
from torchvision.utils import save_image

from model.nst import NST
from model.srres import SRRes
from tg import db_worker
from tg.utils import WEBHOOK_URL_BASE, WEBHOOK_URL_PATH, WEBHOOK_SSL_CERT, WEBHOOK_SSL_PRIV, TELEGRAM_TOKEN, States, \
    WEBHOOK_LISTEN, WEBHOOK_PORT

logger = telebot.logger
telebot.logger.setLevel(logging.INFO)

bot = telebot.AsyncTeleBot(TELEGRAM_TOKEN)

app = web.Application()

apihelper.proxy = {
    'https': 'socks5://student:TH8FwlMMwWvbJF8FYcq0@178.128.203.1:1080'}


async def handle(request: Request):
    if request.match_info.get('token') == bot.token:
        request_body_dict = await request.json()
        update = telebot.types.Update.de_json(request_body_dict)
        bot.process_new_updates([update])
        return web.Response()
    else:
        return web.Response(status=403)


app.router.add_post('/{token}/', handle)


def gce_wrapper(func):
    def wrapper(message: Message):
        try:
            func(message)
        except:
            logger.info('smth went wrong')
            time.sleep(5)
            func(message)

    return wrapper


@bot.message_handler(commands=['start'])
@gce_wrapper
def greet(message: Message):
    bot.reply_to(message, 'Hi there, type one of commands: /nst /srres')
    db_worker.set_state(message.chat.id, States.START.value)


# try:
#     bot.reply_to(message, 'Hi there, type one of commands: /nst /srres')
#     db_worker.set_state(message.chat.id, States.START.value)
# except:
#     logger.info('smth went wrong')
#     time.sleep(5)
#     bot.reply_to(message, 'Hi there, type one of commands: /nst /srres')
#     db_worker.set_state(message.chat.id, States.START.value)


@bot.message_handler(commands=['reset'])
def cmd_reset(message: Message):
    try:
        bot.send_message(message.chat.id, 'Hi there, type one of commands: /nst /srres')
        db_worker.set_state(message.chat.id, States.START.value)
    except:
        logger.info('smth went wrong')
        time.sleep(5)
        bot.send_message(message.chat.id, 'Hi there, type one of commands: /nst /srres')
        db_worker.set_state(message.chat.id, States.START.value)


@bot.message_handler(commands=['help'])
@gce_wrapper
def get_help(message: Message):
    bot.reply_to(message,
                 'Type /nst to start neural style transfer. Then send photo which will be '
                 'used as content. In a very next message send photo which will be used as style. You can '
                 'also request for size of result within second message. Just type it like 384. (default is 256).')


@bot.message_handler(commands=['nst'])
def start_nst(message: Message):
    try:
        bot.reply_to(message, 'Now send me content photo')
        db_worker.set_state(message.chat.id, States.ENTER_FIRST_PIC.value)
    except:
        logger.info('smth went wrong')
        time.sleep(5)
        bot.reply_to(message, 'Now send me content photo')
        db_worker.set_state(message.chat.id, States.ENTER_FIRST_PIC.value)


@bot.message_handler(commands=['srres'])
def start_srres(message: Message):
    try:
        bot.reply_to(message, 'Now send me low res photo')
        db_worker.set_state(message.chat.id, States.ENTER_SRRES_PHOTO.value)
    except:
        logger.info('smth went wrong')
        time.sleep(5)
        bot.reply_to(message, 'Now send me low res photo')
        db_worker.set_state(message.chat.id, States.ENTER_SRRES_PHOTO.value)


@bot.message_handler(
    func=lambda message: db_worker.get_current_state(message.chat.id) == States.ENTER_SRRES_PHOTO.value,
    content_types=['photo'])
def get_low_res(message: Message):
    try:
        file = bot.get_file(message.photo[-1].file_id).wait()
        downloaded_file = bot.download_file(file.file_path).wait()

        with open(f'./images/low_res{message.chat.id}.jpg', 'wb') as file:
            file.write(downloaded_file)

        model = SRRes()

        res = model.transform(f'./images/low_res{message.chat.id}.jpg')
        save_image(res, f'./images/res{message.chat.id}.jpg', normalize=True)
        # res.save(f'./images/res{message.chat.id}.jpg')
        bot.send_photo(message.chat.id, open(f'./images/res{message.chat.id}.jpg', 'rb'))
        db_worker.set_state(message.chat.id, States.START.value)

        for file in ['low_res', 'res']:
            if os.path.exists(f'./images{file}{message.chat.id}.jpg'):
                os.remove(f'./images{file}{message.chat.id}.jpg')

    except:
        logger.info('smth went wrong')
        time.sleep(5)

        file = bot.get_file(message.photo[-1].file_id).wait()
        downloaded_file = bot.download_file(file.file_path).wait()

        with open(f'./images/low_res{message.chat.id}.jpg', 'wb') as file:
            file.write(downloaded_file)

        model = SRRes()

        res = model.transform(f'./images/low_res{message.chat.id}.jpg')
        save_image(res, f'./images/res{message.chat.id}.jpg', normalize=True)
        # res.save(f'./images/res{message.chat.id}.jpg')
        bot.send_photo(message.chat.id, open(f'./images/res{message.chat.id}.jpg', 'rb'))
        db_worker.set_state(message.chat.id, States.START.value)

        for file in ['low_res', 'res']:
            if os.path.exists(f'./images{file}{message.chat.id}.jpg'):
                os.remove(f'./images{file}{message.chat.id}.jpg')


@bot.message_handler(func=lambda message: db_worker.get_current_state(message.chat.id) == States.ENTER_FIRST_PIC.value,
                     content_types=['photo'])
def get_content(message: Message):
    try:
        file = bot.get_file(message.photo[-1].file_id).wait()
        downloaded_file = bot.download_file(file.file_path).wait()

        with open(f'./images/content{message.chat.id}.jpg', 'wb') as file:
            file.write(downloaded_file)

        bot.send_message(message.chat.id, 'Now send me style photo. If you want to specify '
                                          'size send it within your style photo. '
                                          'Size should be less than 512.')
        db_worker.set_state(message.chat.id, States.ENTER_SECOND_PIC.value)
    except:
        logger.info('smth went wrong')
        time.sleep(5)

        file = bot.get_file(message.photo[-1].file_id).wait()
        downloaded_file = bot.download_file(file.file_path).wait()

        with open(f'./images/content{message.chat.id}.jpg', 'wb') as file:
            file.write(downloaded_file)

        bot.send_message(message.chat.id,
                         'Now send me style photo. If you want to specify size send it within your style photo. '
                         'Size should be less than 512.')
        db_worker.set_state(message.chat.id, States.ENTER_SECOND_PIC.value)


@bot.message_handler(func=lambda message: db_worker.get_current_state(message.chat.id) == States.ENTER_SECOND_PIC.value,
                     content_types=['photo'])
def get_style(message: Message):
    try:
        file = bot.get_file(message.photo[-1].file_id).wait()
        downloaded_file = bot.download_file(file.file_path).wait()

        with open(f'./images/style{message.chat.id}.jpg', 'wb') as file:
            file.write(downloaded_file)

        if message.caption:
            if 1 <= int(message.caption) <= 512:
                model = NST(int(message.caption))
            else:
                model = NST(256)
        else:
            model = NST(256)
        res = model.transform(f'./images/content{message.chat.id}.jpg', f'./images/style{message.chat.id}.jpg')
        save_image(res, f'./images/res{message.chat.id}.jpg', normalize=True)
        # model.unload(res).save(f'./images/res{message.chat.id}.jpg')

        bot.send_photo(message.chat.id, open(f'./images/res{message.chat.id}.jpg', 'rb'))
        db_worker.set_state(message.chat.id, States.START.value)

        for file in ['content', 'style', 'res']:
            if os.path.exists(f'./images{file}{message.chat.id}.jpg'):
                os.remove(f'./images{file}{message.chat.id}.jpg')

    except:
        logger.info('smth went wrong')
        time.sleep(5)
        file = bot.get_file(message.photo[-1].file_id).wait()
        downloaded_file = bot.download_file(file.file_path).wait()

        with open(f'./images/style{message.chat.id}.jpg', 'wb') as file:
            file.write(downloaded_file)

        model = NST(128)
        res = model.transform(f'./images/content{message.chat.id}.jpg', f'./images/style{message.chat.id}.jpg')
        model.unload(res).save(f'./images/res{message.chat.id}.jpg')

        bot.send_photo(message.chat.id, open(f'./images/res{message.chat.id}.jpg', 'rb'))
        db_worker.set_state(message.chat.id, States.START.value)

        for file in ['content', 'style', 'res']:
            if os.path.exists(f'./images{file}{message.chat.id}.jpg'):
                os.remove(f'./images{file}{message.chat.id}.jpg')


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
