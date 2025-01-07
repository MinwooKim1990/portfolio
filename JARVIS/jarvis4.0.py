import discord
import openai
from openai import OpenAI
import csv
import re
import requests
import json
from llama_cpp import Llama
import asyncio
import time
import io
from google.oauth2 import service_account
from google.cloud import speech
import os
import sys
from google.cloud import texttospeech_v1
from datetime import datetime
from object_detection_yolo import OD
from dense_image_captioning import img_cap
from OCR import ocr_with_easyocr
from PIL import Image
import random
import pytz
from datetime import datetime
import urllib.request
import aiohttp
import aiofiles

korea_tz = pytz.timezone('Asia/Seoul')
korea_time = datetime.now(korea_tz)

class JarvisBot:
    def __init__(self):
        self.intents = discord.Intents.default()
        self.intents.message_content = True
        self.client = discord.Client(intents=self.intents)
        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.setup_credentials()
        self.setup_clients()

    def setup_credentials(self):
        self.SERVER_ID = os.environ["SERVER_ID"]
        self.CHANNEL_ID = os.environ["CHANNEL_ID"]
        self.papago_client_id = os.environ["PAPAGO_CLIENT_ID"]
        self.papago_client_secret = os.environ["PAPAGO_CLIENT_SECRET"]
        self.client_file = os.environ["SPEECH_TO_TEXT_KEY"]
        self.credentials = service_account.Credentials.from_service_account_file(self.client_file)

    def setup_clients(self):
        self.stt_client = speech.SpeechClient(credentials=self.credentials)
        self.model_path = "llama/llama-2-13b-chat/ggml-model-f16.gguf"
        self.llama_model = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_gpu_layers=4,
            use_mlock=True
        )

    async def GPT4(self, messages, max_tokens=200):
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4-1106-preview",
                messages=messages,
                temperature=0.8,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in GPT4: {e}")
            return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."

    async def LLAMA(self, messages, token=200, temp=0.8):
        try:
            response = await asyncio.to_thread(
                self.llama_model,
                messages,
                max_tokens=token,
                temperature=temp
            )
            return response["choices"][0]["text"]
        except Exception as e:
            print(f"Error in LLAMA: {e}")
            return "LLAMA 모델 처리 중 오류가 발생했습니다."

    async def lang_trans(self, input_string, source, target):
        try:
            url = "https://openapi.naver.com/v1/papago/n2mt"
            headers = {
                "X-Naver-Client-Id": self.papago_client_id,
                "X-Naver-Client-Secret": self.papago_client_secret
            }
            data = {
                "source": source,
                "target": target,
                "text": input_string
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=data) as response:
                    result = await response.json()
                    return result["message"]["result"]["translatedText"]
        except Exception as e:
            print(f"Translation error: {e}")
            return "번역 중 오류가 발생했습니다."

    async def speech_to_text(self, file):
        try:
            async with aiofiles.open(file, 'rb') as f:
                content = await f.read()
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
                sample_rate_hertz=48000,
                language_code='ko-KR',
            )
            response = await asyncio.to_thread(
                self.stt_client.recognize,
                config=config,
                audio=audio
            )
            return " ".join(result.alternatives[0].transcript for result in response.results)
        except Exception as e:
            print(f"STT error: {e}")
            return "음성 인식 중 오류가 발생했습니다."

    async def on_message(self, message):
        if message.author == self.client.user:
            return

        try:
            if message.attachments:
                user_message = 'file attached'
                for attachment in message.attachments:
                    folder = ''
                    if attachment.content_type.startswith('audio'):
                        wait_msg = await message.channel.send("Wait...")
                        folder = 'audio'
                        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                        file_extension = os.path.splitext(attachment.filename)[1]
                        new_file_name = f"{current_time}{file_extension}"
                        file_path = f"./{folder}/{new_file_name}"
                        await attachment.save(file_path)
                        user_message = await self.speech_to_text(get_latest_file(folder, 'audio'))
                        await wait_msg.delete()
                    elif attachment.content_type.startswith('image'):
                        wait_msg = await message.channel.send("Consider the attached image is characters for OCR or image for captioning")
                        folder = 'image'
                        file_extension = os.path.splitext(attachment.filename)[1]
                        new_file_name = f"User_image{file_extension}"
                        temp_file_path = f"./{folder}/temp_{new_file_name}"
                        await attachment.save(temp_file_path)

                        with Image.open(temp_file_path) as img:
                            if img.mode == "RGBA" and file_extension not in ['.jpg', '.jpeg']:
                                img = img.convert("RGB")
                            new_file_name = f"User_image.jpg"
                            final_file_path = f"./{folder}/{new_file_name}"
                            img.save(final_file_path)
                        os.remove(temp_file_path)

                        ocr_input = ocr_with_easyocr('image/User_image.jpg')
                        if ocr_input == []:
                            cate, obj_img, tot_img = OD('image/User_image.jpg')
                            ic = img_cap(cate, tot_img, obj_img)
                        image_analysis_switch = True
                        await wait_msg.delete()
            else:
                user_message = str(message.content)

            print(message.author.name + ' said ' + user_message.lower() + ' in ' + message.channel.name)

            if message.channel.name == 'jarvis':
                thinking_msg = await message.channel.send("Thinking...")
                retries = 0
                while retries < 3:
                    try:
                        speak_prefixes = ['/speech', '/spk', '/speak', '/말해', '/s']
                        llama_prefixes = ['/lama', '/llama', '/라마']
                        img_gen_prefixes = ['/gen', '/image', '/img', '/이미지', '/생성']
                        voice_prefixes = ['/voice', '/목소리']
                        trans_prefixes = ['/trans', '/번역', '/tr']

                        if any(prefix in user_message for prefix in voice_prefixes):
                            filtered_message = user_message
                            for prefix in voice_prefixes:
                                if prefix in filtered_message:
                                    user_message = filtered_message.replace(prefix, '', 1).strip()
                                    break
                            if user_message == '':
                                skip = True
                                output = user_message
                                c_time = 0
                        if any(prefix in user_message for prefix in speak_prefixes):
                            speech_switch = True
                            filtered_message = user_message
                            for prefix in llama_prefixes:
                                if prefix in filtered_message:
                                    user_message = filtered_message.replace(prefix, '', 1).strip()
                        if '/r' in user_message:
                            user_message = extract_patterns(user_message)
                            remember(extract_patterns(user_message))
                        if any(prefix in user_message for prefix in llama_prefixes):
                            filtered_message = user_message
                            for prefix in llama_prefixes:
                                if prefix in filtered_message:
                                    filtered_message = filtered_message.replace(prefix, '', 1).strip()
                                    break
                            msgs = await message.channel.send("LLAMA2 Computing Start")
                            s_time = time.time()
                            output = await self.LLAMA(filtered_message, 400, 0.8)
                            e_time = time.time()
                            c_time = e_time - s_time
                            await msgs.delete()
                        elif any(prefix in user_message for prefix in img_gen_prefixes):
                            filtered_message = user_message
                            for prefix in img_gen_prefixes:
                                if prefix in filtered_message:
                                    filtered_message = filtered_message.replace(prefix, '', 1).strip()
                                    break
                            img_gen_switch = True
                            if any([ord(char) >= 0x0041 and ord(char) <= 0x007A for char in filtered_message]) or \
                                    any([ord(char) >= 0x0061 and ord(char) <= 0x007A for char in filtered_message]):
                                filtered_message = filtered_message
                            else:
                                filtered_message = await self.GPT4([{"role": "user", "content": f"Translate input prompt and write image generation prompt into Enlglish. Output with only generated string only.: {filtered_message}"}])
                            msgs = await message.channel.send("Image Generation Start")
                            s_time = time.time()
                            output = image_generation(filtered_message)
                            e_time = time.time()
                            c_time = e_time - s_time
                            user_message = f'Image Generation prompt: {filtered_message}'
                            await msgs.delete()
                        elif image_analysis_switch:
                            msgs = await message.channel.send("Image Analysis Start")
                            s_time = time.time()
                            if ocr_input != []:
                                output = f'-Detected texts: {ocr_input}'
                            else:
                                output = await self.GPT4([{"role": "user", "content": f"Complete a detailed image captioning without mention of provided captioned result of the image using the image captioning results provided as a list which captioned the entire image, and cropped images of each detected object. Consider that object detection and image captioning results are not always correct, so please complete the captioning that is most correct and Please Do Not mention about the result of object detection, image captioning also do not use negative words. Image captioning sentences:{ic}"}])
                                output = f'-Simple image captioning: {ic_extra} \n\n-AI Thinking image captioning: {output}'
                            e_time = time.time()
                            c_time = e_time - s_time
                            image_analysis_switch = False
                            await msgs.delete()
                        elif any(prefix in user_message for prefix in trans_prefixes):
                            filtered_message = user_message
                            for prefix in trans_prefixes:
                                if prefix in filtered_message:
                                    user_message = filtered_message.replace(prefix, '', 1).strip()
                            s_time = time.time()
                            output = await self.lang_trans(user_message, 'auto', 'ko')
                            e_time = time.time()
                            c_time = e_time - s_time
                        else:
                            if skip:
                                await message.channel.send('Skip Computing')
                            else:
                                msgs = await message.channel.send("GPT4 Computing Start")
                                s_time = time.time()
                                messages_for_api = prompt_to_chat(message.author.name, user_message)
                                response = await self.GPT4(messages_for_api)
                                output = response
                                e_time = time.time()
                                c_time = e_time - s_time
                                await msgs.delete()

                        rethink_words = ["I don't know", "not sure", "my latest training data", "I can't", "current knowledge", "up to December", "real-time capabilities", "As of my knowledge", "updated till", "October 2021", "real-time", "cut-off", 'as an AI']
                        if any(prefix in output for prefix in rethink_words):
                            msgs = await message.channel.send("Searching in the Google")
                            keywords = await self.GPT4([{"role": "user", "content": f"Extract Keywords for google search from this prompt: {user_message}"}])
                            google_search_res = google_search(keywords)
                            combined_res = '\n\n'.join(google_search_res)
                            await msgs.delete()
                            output = f"I searched keywords on the Google. \n\nBased on a Google search ({keywords}). \n\nCheck out summarized result or relevant link below: \n{combined_res}"

                        if speech_switch:
                            sample = output[:20]
                            if any([ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in sample]):
                                language = "ko-KR"
                                if current_value % 2 == 0:
                                    Name = 'ko-KR-Standard-A'
                                    gender = 'FEMALE'
                                else:
                                    Name = 'ko-KR-Standard-C'
                                    gender = 'MALE'
                            elif any([ord(char) >= 0x0041 and ord(char) <= 0x007A for char in sample]) or \
                                    any([ord(char) >= 0x0061 and ord(char) <= 0x007A for char in sample]):
                                language = "en-GB"
                                if current_value % 2 == 0:
                                    Name = 'en-GB-News-G'
                                    gender = 'FEMALE'
                                else:
                                    Name = 'en-GB-News-J'
                                    gender = 'MALE'
                            else:
                                language = "ja-JP"
                                if current_value % 2 == 0:
                                    Name = 'ja-JP-Standard-A'
                                    gender = 'FEMALE'
                                else:
                                    Name = 'ja-JP-Standard-C'
                                    gender = 'MALE'
                            text_to_speech(output, language, Name, gender)
                            try:
                                await message.channel.send(file=discord.File(get_latest_file('speech', 'audio')))
                            except Exception as e:
                                print(f"Error: {e}")
                            speech_switch = False

                        if img_gen_switch:
                            output = output
                            await message.channel.send(f"Image prompt: {filtered_message}")

                        if len(output) > 1200:
                            split_output = split_message(output)
                            for msg in split_output:
                                await message.channel.send(msg)
                        else:
                            await message.channel.send(output)
                        await message.channel.send(f"-(The code ran for {c_time:.2f} seconds.)-")
                        add_history(message.author.name, user_message, output)
                        await thinking_msg.delete()
                        break

                    except openai.error.OpenAIError as e:
                        print(f"OpenAI error: {e}")
                        retries += 1
                        await asyncio.sleep(1)

                if retries == 3:
                    await thinking_msg.delete()
                    await message.channel.send("I encountered an error multiple times. Please try again later.")

        except Exception as e:
            print(f"Error processing message: {e}")
            await message.channel.send("명령어 처리 중 오류가 발생했습니다.")

    def run(self):
        @self.client.event
        async def on_ready():
            print(f'{self.client.user} has connected to Discord!')

        @self.client.event
        async def on_message(message):
            await self.on_message(message)

        self.client.run(os.environ["DISCORD_BOT_TOKEN"], reconnect=True)

if __name__ == "__main__":
    bot = JarvisBot()
    bot.run()