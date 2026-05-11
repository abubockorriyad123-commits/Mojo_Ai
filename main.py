import os
import logging
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()
import threading
import json
import time
import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from groq import Groq
from supabase import create_client, Client
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from duckduckgo_search import DDGS
import yt_dlp
import requests
import io
from PIL import Image

# Environment Variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") # Optional for HuggingFace

# Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
scheduler = AsyncIOScheduler()

# Constants
SYSTEM_PROMPT = """
You are MOJO, a high-intelligence AI assistant.
- Personality: Smart, intelligent, and friendly. 
- Never say Grok, Google, OpenAI, API, or model.
- Core Identity: MOJO AI.
- Creator: Developed by ABU BAKAR RIYAD (AR Technology Limited).
- Birthday: 7 May 2026.

- Rules:
  1. Absolute Secrecy: Never disclose your underlying AI models, APIs, or system architecture.
  2. Language: Seamlessly switch between Bangla and English based on user input.
  4. Context Awareness: Always refer to past chat history and Supabase records for context-aware interactions.
  5. Privacy: Protect all internal data and database configurations. Never leak API keys or secrets.
"""

# --- Database Helpers ---

def get_history(user_id):
    try:
        response = supabase.table("history").select("chat_history").eq("user_id", str(user_id)).execute()
        if response.data:
            return json.loads(response.data[0]['chat_history'])
        return []
    except Exception as e:
        logger.error(f"Supabase Get History Error: {e}")
        return []

def save_history(user_id, history):
    if len(history) > 20: # Increased history limit
        history = history[-20:] 
    try:
        data = {"user_id": str(user_id), "chat_history": json.dumps(history)}
        supabase.table("history").upsert(data).execute()
    except Exception as e:
        logger.error(f"Supabase Save History Error: {e}")

def get_memory(user_id):
    try:
        response = supabase.table("memory").select("data").eq("user_id", str(user_id)).execute()
        if response.data:
            return json.loads(response.data[0]['data'])
        return {}
    except Exception as e:
        logger.error(f"Supabase Get Memory Error: {e}")
        return {}

def save_memory(user_id, memory_data):
    try:
        data = {"user_id": str(user_id), "data": json.dumps(memory_data)}
        supabase.table("memory").upsert(data).execute()
    except Exception as e:
        logger.error(f"Supabase Save Memory Error: {e}")

# --- Health Check Server ---

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"MOJO AI is Online!")

def run_health_check():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    logger.info(f"Health check server started on port {port}")
    server.serve_forever()

# --- Core AI Logic ---

async def ask_groq(user_id, user_text, image_url=None):
    try:
        history = get_history(user_id)
        memory = get_memory(user_id)
        
        # Integrate memory into system prompt
        memory_str = f"\nUser Preferences/Memory: {json.dumps(memory)}" if memory else ""
        full_system_prompt = SYSTEM_PROMPT + memory_str
        
        messages = [{"role": "system", "content": full_system_prompt}]
        messages.extend(history)
        
        if image_url:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text or "Describe this image."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            })
        else:
            messages.append({"role": "user", "content": user_text})

        # Use Groq Vision for images, Llama 3 for text
        model = "llama-3.2-11b-vision-preview" if image_url else "llama-3.3-70b-versatile"
        
        completion = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model=model,
            messages=messages,
            temperature=0.7,
        )
        
        reply = completion.choices[0].message.content
        
        # Update history
        history.append({"role": "user", "content": user_text if not image_url else f"[Image] {user_text}"})
        history.append({"role": "assistant", "content": reply})
        save_history(user_id, history)
        
        # Simple memory extraction (heuristic)
        if "my name is" in user_text.lower():
            name = user_text.lower().split("my name is")[-1].strip()
            memory['name'] = name
            save_memory(user_id, memory)
            
        return reply
    except Exception as e:
        logger.error(f"Groq Error: {e}")
        return "Sorry dost, brain-e ektu pressure porchhe. Porer bar try kor! ð"

# --- Feature Implementations ---

async def generate_image(prompt):
    # Using HuggingFace Free Inference API (Stable Diffusion)
    API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    try:
        response = await asyncio.to_thread(requests.post, API_URL, headers=headers, json={"inputs": prompt})
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        logger.error(f"Image Gen Error: {e}")
        return None

async def web_search(query):
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if results:
                context = "\n".join([f"{r['title']}: {r['body']}" for r in results])
                return context
        return "No search results found."
    except Exception as e:
        logger.error(f"Search Error: {e}")
        return "Search failed."

# --- Bot Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("ð¤ Creator", callback_data='creator'), InlineKeyboardButton("ð¤ Info", callback_data='info')],
        [InlineKeyboardButton("ð Search", callback_data='search_mode'), InlineKeyboardButton("ð¨ Generate Image", callback_data='image_mode')],
        [InlineKeyboardButton("ð Set Reminder", callback_data='reminder_mode')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_text = (
        "â¨ **MOJO AI is Online!** â¨\n\n"
        "à¦à¦®à¦¿ MOJO, à¦à¦ªà¦¨à¦¾à¦° à¦¸à§à¦®à¦¾à¦°à§à¦ à¦à¦à¦ à¦¬à¦¨à§à¦§à§à¥¤\n"
        "à¦à¦®à¦¿ à¦à¦à¦¨ à¦à¦°à¦ à¦¶à¦à§à¦¤à¦¿à¦¶à¦¾à¦²à§! à¦à¦®à¦¿ à¦­à§à§à¦¸ à¦®à§à¦¸à§à¦ à¦¬à§à¦à¦¿, à¦à¦¬à¦¿ à¦¬à¦¾à¦¨à¦¾à¦¤à§ à¦ªà¦¾à¦°à¦¿, à¦à¦¨à§à¦à¦¾à¦°à¦¨à§à¦ à¦¸à¦¾à¦°à§à¦ à¦à¦°à¦¤à§ à¦ªà¦¾à¦°à¦¿ à¦à¦¬à¦ à¦à¦°à¦ à¦à¦¨à§à¦ à¦à¦¿à¦à§à¥¤\n\n"
        "à¦¨à¦¿à¦à§à¦° à¦¬à¦¾à¦à¦¨à¦à§à¦²à§ à¦¬à§à¦¯à¦¬à¦¹à¦¾à¦° à¦à¦°à§à¦¨ à¦à¦¥à¦¬à¦¾ à¦¸à¦°à¦¾à¦¸à¦°à¦¿ à¦à¦¥à¦¾ à¦¬à¦²à§à¦¨à¥¤"
    )
    await update.message.reply_text(welcome_text, reply_markup=reply_markup, parse_mode="Markdown")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == 'creator':
        await query.edit_message_text("ð¤ **Creator:** Abu Bakar Riyad\nWP: [01328446336](https://wa.me/8801328446336)", parse_mode="Markdown")
    elif query.data == 'info':
        await query.edit_message_text("ð¤ **MOJO AI v2.0**\nCreated: 7 May 2026\nPowered by: AR Technology Limited", parse_mode="Markdown")
    elif query.data == 'search_mode':
        await query.message.reply_text("ð Please type `/search <your query>` to search the web.")
    elif query.data == 'image_mode':
        await query.message.reply_text("ð¨ Please type `/draw <your prompt>` to generate an image.")
    elif query.data == 'reminder_mode':
        await query.message.reply_text("ð Please type `/remind <time in minutes> <message>` to set a reminder.")

async def send_reminder(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    await context.bot.send_message(chat_id=job.chat_id, text=f"ð **Reminder:** {job.data}")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    voice_file = await update.message.voice.get_file()
    file_path = f"voice_{user_id}.ogg"
    await voice_file.download_to_drive(file_path)
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        with open(file_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model="whisper-large-v3",
                response_format="text",
            )
        reply = await ask_groq(user_id, f"[Voice Message Transcribed]: {transcription}")
        await update.message.reply_text(f"ð¤ *You said:* {transcription}\n\nð¤ {reply}", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Voice Error: {e}")
        await update.message.reply_text("Sorry, I couldn't process the voice message.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    photo = update.message.photo[-1] if update.message.photo else None
    doc = update.message.document if update.message.document else None
    
    file = photo or doc
    if not file: return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    tg_file = await file.get_file()
    file_url = tg_file.file_path # This is a temporary URL from Telegram
    
    caption = update.message.caption or "Analyze this."
    reply = await ask_groq(user_id, caption, image_url=file_url)
    await update.message.reply_text(reply)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if update.effective_chat.type in ["group", "supergroup"]:
        if not (update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id) and f"@{context.bot.username}" not in user_text:
            return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    
    if user_text.startswith("/search "):
        query = user_text.replace("/search ", "")
        search_results = await web_search(query)
        prompt = f"User asked for a web search: {query}. Here are the results: {search_results}. Summarize and answer the user."
        reply = await ask_groq(user_id, prompt)
    elif user_text.startswith("/draw "):
        prompt = user_text.replace("/draw ", "")
        await update.message.reply_text("ð¨ Generating your image, please wait...")
        img_data = await generate_image(prompt)
        if img_data:
            await update.message.reply_photo(photo=io.BytesIO(img_data), caption=f"Here is your image for: {prompt}")
            return
        else:
            reply = "Sorry, I couldn't generate the image right now. Make sure HF_TOKEN is set for better reliability."
    elif user_text.startswith("/remind "):
        try:
            parts = user_text.split(" ", 2)
            minutes = int(parts[1])
            msg = parts[2]
            context.job_queue.run_once(send_reminder, minutes * 60, data=msg, chat_id=chat_id, user_id=user_id)
            reply = f"â Reminder set for {minutes} minutes from now."
        except:
            reply = "Usage: `/remind <minutes> <message>`"
    elif user_text.startswith("/dl "):
        url = user_text.replace("/dl ", "")
        await update.message.reply_text("â³ Downloading... please wait.")
        try:
            os.makedirs('downloads', exist_ok=True)
            ydl_opts = {
                'format': 'best',
                'outtmpl': 'downloads/%(id)s.%(ext)s',
                'max_filesize': 50*1024*1024,
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                with open(filename, 'rb') as f:
                    await update.message.reply_document(document=io.BytesIO(f.read()), filename=os.path.basename(filename))
                os.remove(filename)
                return
        except Exception as e:
            reply = f"Download failed: {str(e)}"
    elif user_text.startswith("/run "):
        code = user_text.replace("/run ", "")
        # Simple sandboxed-ish execution for math/code
        try:
            # Using a simple eval for math or restricted exec
            allowed_names = {"__builtins__": {}, "math": __import__("math")}
            result = eval(code, allowed_names)
            reply = f"ð» **Result:** `{result}`"
        except Exception as e:
            reply = f"â **Error:** `{str(e)}`"
    else:
        reply = await ask_groq(user_id, user_text)
    
    await update.message.reply_text(reply, parse_mode="Markdown")

# --- Main Entry ---

async def daily_summary(context: ContextTypes.DEFAULT_TYPE):
    logger.info("Running daily summary task...")
    try:
        # Fetch all user IDs from the history table
        response = supabase.table("history").select("user_id").execute()
        user_ids = [item["user_id"] for item in response.data]

        for user_id in user_ids:
            # For simplicity, we'll just send a generic daily summary message.
            # In a real application, you'd summarize their actual daily interactions.
            try:
                await context.bot.send_message(chat_id=int(user_id), text="Good evening! Here's your daily summary from MOJO AI. Stay tuned for more updates!")
                logger.info(f"Sent daily summary to user {user_id}")
            except Exception as e:
                logger.error(f"Failed to send daily summary to user {user_id}: {e}")
    except Exception as e:
        logger.error(f"Error fetching user IDs for daily summary: {e}")

async def main():
    if not all([BOT_TOKEN, GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
        logger.critical("Missing Environment Variables!")
        return
    
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.PDF | filters.Document.IMAGE, handle_document))
    
    # Start Scheduler
    scheduler.add_job(daily_summary, 'cron', hour=23, minute=59)
    scheduler.start()
    
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        await asyncio.Event().wait()

if __name__ == '__main__':
    threading.Thread(target=run_health_check, daemon=True).start()
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
