import json
from telegram import Bot, InputFile, ParseMode
from telegram.error import NetworkError

TELEGRAM_BOT_KEY = '929296871:AAELba3uA0XgTR0Gh0WkeSKFy3CLBGmsiKs' # ibp_apps_bot
TELEGRAM_CHAT_ID = 713375966 # rjanuzi

def send_simple_message(text, bot_key=TELEGRAM_BOT_KEY, chat_id=TELEGRAM_CHAT_ID):
    try:
        Bot(token=bot_key).send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)
        return True
    except:
        return False

def send_img(img_path, bot_key=TELEGRAM_BOT_KEY, chat_id=TELEGRAM_CHAT_ID, caption=''):
    try:
        Bot(token=bot_key).send_photo(chat_id=chat_id, photo=InputFile(open(img_path, 'rb')), caption=caption)
        return True
    except:
        return False

def send_document(document_path, bot_key=TELEGRAM_BOT_KEY, chat_id=TELEGRAM_CHAT_ID, caption=''):
    try:
        Bot(token=bot_key).send_document(chat_id=chat_id, document=InputFile(open(document_path, 'rb')), caption=caption)
        return True
    except:
        return False

def get_messages(bot_key=TELEGRAM_BOT_KEY):
    # Get offset
    try:
        offset_file = open(r'telegram_messages_offset.json', 'r')
        offset = json.load(fp=offset_file)
        offset_file.close()
    except FileNotFoundError:
        offset = {'offset':0}

    bot = Bot(token=bot_key)
    updates = bot.get_updates(offset=offset['offset'])
    updates = list(filter(lambda u: u and u.message, updates)) # Eliminates None
    messages = [
                {'chat_id': u.message.chat_id,
                 'user_id': u.message.from_user.id,
                 'message': u.message.text} for u in updates
               ]

    if len(updates) > 0:
        offset['offset'] = updates[-1].update_id+1

    offset_file = open(r'telegram_messages_offset.json', 'w', encoding="utf-8")
    json.dump(obj=offset, fp=offset_file)
    offset_file.close()

    return messages
