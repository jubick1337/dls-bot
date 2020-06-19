import logging

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor

from model.nst import NST

TELEGRAM_TOKEN = '1109017372:AAG7i7w4mDWlY9oUqaDQW8rN2NM105OODZ4'

bot = Bot(token=TELEGRAM_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


# States
class Form(StatesGroup):
    command = State()
    content = State()
    style = State()


@dp.message_handler(commands='start')
async def handle_start(message: types.Message):
    await Form.command.set()
    return await message.reply("Hi there! Use one of the commands: /nst")


@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    """
    Allow user to cancel any action
    """
    current_state = await state.get_state()
    if current_state is None:
        return

    logging.info('Cancelling state %r', current_state)
    # Cancel state and inform user about it
    await state.finish()
    # And remove keyboard (just in case)
    return await message.reply('Cancelled.')


@dp.message_handler(state=Form.command)
async def process_command(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['command'] = message.text

    await Form.next()
    return await message.reply("Now send content photo")


@dp.message_handler(content_types=['photo', 'document'], state=Form.content)
async def process_content(message: types.Message):
    await Form.next()
    if message.photo:
        await message.photo[-1].download(f'content{message.chat.id}.jpg')
    else:
        await message.document.download(f'content{message.chat.id}.jpg')

    return await message.reply("Now send a style photo")


@dp.message_handler(lambda message: len(message.photo) != 1, state=Form.content)
async def process_content_invalid(message: types.Message):
    return await message.reply("I was waiting for photo")


@dp.message_handler(lambda message: not message.photo, state=Form.style)
async def process_style_invalid(message: types.Message):
    return await message.reply("I was waiting for photo")


@dp.message_handler(content_types=['photo', 'document'], state=Form.style)
async def get_result(message: types.Message, state: FSMContext):
    if message.photo:
        await message.photo[-1].download(f'style{message.chat.id}.jpg')
        print('downloaded')
    else:
        await message.document.download(f'style{message.chat.id}.jpg')
        print('downloaded')

    model = NST(128)
    await message.answer('now wait a little')
    print('model created')
    res = await model.transform(f'content{message.chat.id}.jpg', f'style{message.chat.id}.jpg')
    model.unload(res).save(f'res{message.chat.id}.jpg')
    print('transfer done')
    await state.finish()
    return await message.answer_photo(types.InputFile(f'res{message.chat.id}.jpg'))


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
