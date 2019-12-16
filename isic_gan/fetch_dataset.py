from dataset import download_imgs
from _telegram import send_simple_message

download_imgs(2000)

send_simple_message(text='Imgs fetching concluded.')
