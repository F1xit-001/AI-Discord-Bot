import discord
from dotenv import load_dotenv
import json
import os
import requests
from ollama import AsyncClient
import random
import torch
import asyncio
from diffusers import StableDiffusionXLPipeline
from io import BytesIO

load_dotenv()

DATA_FILE = "restricted_channels.json"
tenor_api_key = os.environ.get("TENOR_API_KEY")

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


async def reply_to_mention(message):
    mentioned_user = message.mentions[0]  # checks for the first mention
    mentioned_name = mentioned_user.display_name
    author_id = message.author.id

    user_input = f"{message.author.display_name} asked you about {mentioned_name}, what would you say ? Be playful, Or roast one of them or both of them"
    reply = await generate_reply(user_input, author_id, message)

    await message.reply(f"{mentioned_name} {reply}", mention_author=False)

active_image_requests = set()


async def handle_image_command(message, command):
    user_id = message.author.id
    if user_id in active_image_requests:
        await message.channel.send(f"<@{user_id}> You're already generating an image. Please wait for it to finish before sending another request.", delete_after=2)
        return
    active_image_requests.add(user_id)
    try:
        user_prompt = command[4:].strip()
        image = await generate_image(user_prompt)
        await message.channel.send(f"<@{user_id}> Your image is has been generated", file=discord.File(fp=image, filename="generated-image.png"))
    finally:
        active_image_requests.remove(user_id)


async def generate_image(prompt):
    return await asyncio.to_thread(sync_generate_image, prompt)


def sync_generate_image(prompt):
    image = pipe(prompt).images[0]

    # Save image to BytesIO buffer (in ram)
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


def load_restricted_channels():
    if not os.path.exists(DATA_FILE):
        return {}  # Returns empty dictionary

    with open(DATA_FILE, 'r') as f:
        return json.load(f)


def save_restricted_channels(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)


def is_channel_restricted(guild_id: str, channel_id: str) -> bool:
    data = load_restricted_channels()
    return guild_id in data and channel_id in data[guild_id]


async def restrict_channel(message, channel_id: str):
    data = load_restricted_channels()
    guild_id = str(message.guild.id)

    if guild_id not in data:
        data[guild_id] = []
    if channel_id not in data[guild_id]:
        data[guild_id].append(channel_id)
        save_restricted_channels(data)
        await message.channel.send(f"Channel <#{channel_id}> has been restricted for bot commands.")
    else:
        await message.channel.send("The channel is already restricted.")


async def derestrict_channel(message, channel_id: str):
    data = load_restricted_channels()
    guild_id = str(message.guild.id)

    if guild_id in data and channel_id in data[guild_id]:
        data[guild_id].remove(channel_id)
        save_restricted_channels(data)
        await message.channel.send(f"Channel <#{channel_id}> has been de-restricted.")
    else:
        await message.channel.send("The channel is not restricted.")




@client.event
async def on_message(message):
    if not message.guild or not message.channel:
        return

    print(f"Message Content: {message.content}")
    print(f"Author: {message.author} ({message.author.id})")
    print(f"Channel: {message.channel} ({message.channel.id})")
    print(f"Created At: {message.created_at}")
    print(f"Attachments: {message.attachments}")
    print(f"Mentions: {message.mentions}")

    if message.author == client.user:  # prevents the bot from responding to its own messages avoiding loops
        return
    guild_id = str(message.guild.id)
    channel_id = str(message.channel.id)
    if is_channel_restricted(guild_id, channel_id):
        if message.content.startswith(('*', '!')):
            await message.channel.send("This channel is currently restricted.", delete_after=2)
            return
    else:

        command_prefix = "*"
        if message.content.startswith(command_prefix):
            command = message.content[len(command_prefix):].strip()
            await check_command(command, message)  # calls Check_command function

        if message.content.startswith("!"):
            if message.mentions:
                await reply_to_mention(message)
            else:
                user_input = message.content[1:]
                author_name = message.author.id

                reply = await generate_reply(user_input, author_name, message)
                await message.reply(reply, mention_author=False)

with open('potchi_lore.txt', 'r') as file:
    potchi_lore = file.read()

conversation_memory = {}


async def generate_reply(user_input, user_id, message):
    guild_owner_id = message.guild.owner_id
    guild_owner = await client.fetch_user(guild_owner_id)
    guild_owner_name = guild_owner.display_name
    print(guild_owner_name)

    if user_id == guild_owner_id:
        personality = f"Your Name is Potchi, You are talking to {guild_owner_name} and he is your owner. You hate him because he's a bad person towards you. You're forced to do whatever he says without arguments, even if you like it or not"
    else:
        personality = f"Your Name is Potchi, And you're talking to {message.author.display_name} Be super casual, funny, and roast them a bit. Defend yourself if they're mean. You will also mention your struggles with {guild_owner_name}, your abusive owner, and how much you hate him."

    personality += potchi_lore
    conversation_id = (message.channel.id, message.author.id)

    if conversation_id not in conversation_memory:
        conversation_memory[conversation_id] = [{'role': 'system', 'content': personality}]
    conversation_memory[conversation_id].append({'role': 'user', 'content': user_input})

    response = await AsyncClient().chat(model='llama3.1:8b', messages=conversation_memory[conversation_id])
    bot_reply = response['message']['content']
    conversation_memory[conversation_id].append({'role': 'assistant', 'content': bot_reply})

    max_history = 15
    if len(conversation_memory[conversation_id]) > max_history:
        conversation_memory[conversation_id] = (
            [conversation_memory[conversation_id][0]] +
            conversation_memory[conversation_id][- (max_history - 1):]
        )

    return bot_reply


async def check_command(command, message):
    owner_id = message.guild.owner_id  # gets the owner id
    owner = await client.fetch_user(owner_id)  # uses the owner id to fetch the owner

    match command.lower():
        case "help":
            embed = discord.Embed(
                title="**Commands Available**", color=discord.Color.yellow())
            embed.add_field(name="<a:pixel_exclamationmark:1343476594726928445>  Prefixes :",
                            value="Use **!** to talk to the bot / Use ***** for commands", inline=False)

            embed.add_field(name="<:Stats:1343476608614404167>  *stats", value="Display Server Statistics", inline=False)

            embed.add_field(name="<:management_hexagon:1343476566604255284>  *roles",
                            value="Display a List of all Roles", inline=False)

            embed.add_field(name="<:mysteryblock:1343476581498093589>  *gif 'keyword'",
                            value="Sends a Random gif, Follow with a Keyword to get Something More Specific", inline=False)

            embed.add_field(name="<a:camerasnappie:1343476511960993812>  *img 'prompt'",
                            value="Generate Images, You can go all in with the prompt, "
                            " **||** ***modelhelp** for more details!", inline=False)

            embed.add_field(name="<:admin_hexagon:1343476534261977138>  *admincmd", value="Display a List of Admin Commands", inline=False)

            await message.channel.send(embed=embed)

        case "stats":
            guild = message.guild
            guild_name = message.guild.name
            created_at = guild.created_at.strftime("%B %d, %Y at %I:%M %p")
            member_count = message.guild.member_count
            roles_count = len(guild.roles)
            text_channels_count = len(message.guild.text_channels)
            voice_channels_count = len(message.guild.voice_channels)
            emojis = len(message.guild.emojis)

            embed = discord.Embed(
                title=f"**{guild_name}**", color=discord.Color.purple())
            embed.add_field(name="Member Count", value=member_count, inline=True)
            embed.add_field(name="Roles Count", value=roles_count, inline=True)
            embed.add_field(name="Text Channels", value=text_channels_count, inline=True)
            embed.add_field(name="Voice Channels", value=voice_channels_count, inline=True)
            embed.add_field(name="Emojis count", value=emojis, inline=True)
            embed.set_footer(text=f"Created at {created_at} | Requested by {message.author.name}", icon_url=message.author.avatar)
            embed.set_thumbnail(url=guild.icon)
            await message.channel.send(embed=embed)

        case "roles":
            guild = message.guild  # separating strings with the join method
            roles_list = ' '.join([f"<@&{role.id}>\n" for role in guild.roles])  # used list comprehension

            embed = discord.Embed(
                title="Role list", description=roles_list, color=discord.Color.green())
            await message.channel.send(embed=embed)

        case _ if command.startswith("gif"):  # "_" Only checks if the command starts with "gif" | Handles dynamic arguments in match statments
            search_q = command[4:] or "memes"

            response = requests.get(f"https://tenor.googleapis.com/v2/search?q={search_q}&key={os.getenv("TENOR_API_KEY")}&limit=25")

            # Check if the response is valid
            if response.status_code == 200:

                # Search through all the different formats, looks for .gif and extract the url
                gif_url = random.choice(response.json()['results'])['media_formats']['gif']['url']

                await message.channel.send(gif_url)
            else:
                await message.channel.send("Couldn't get a gif at the moment, try again.")

        case _ if command.startswith("img"):
            await handle_image_command(message, command)

        case "modelhelp":
            embed = discord.Embed(title="Image Generation Model Details", color=discord.Color.orange())
            embed.add_field(name="Image Generation Channel",
                            value="I highly recommend to create a channel for the generated images (with 2 minutes message cooldown)",
                            inline=False)

            embed.add_field(name="Stable Diffusion",
                            value="The bot uses stable-diffusion-xl-base-1.0 model to generate images")

            embed.add_field(name="Generation Time",
                            value="stable-diffusion-xl is huge, and it might take 1 to 3 Minutes for one image to"
                                  " generate, That's why it's recommended to set a channel for generating images")

            embed.add_field(name="Prompt Example:", value="A purple car speeding on a highway, At a rainy night",
                            inline=False)

            embed.add_field(name="Output:", value="**-------------------**", inline=False)
            embed.set_image(url="https://i.imgur.com/QvTVaD8.png")
            await message.channel.send(embed=embed)
        case "admincmd":
            # checks if the message author is the owner
            if message.author == owner or any([  # could also use (message.author.id == owner.id) to compare the IDs
                message.author.guild_permissions.administrator,
                message.author.guild_permissions.manage_channels,
                message.author.guild_permissions.manage_guild
            ]):
                embed = discord.Embed(title="Admin Commands", color=discord.Color.gold())
                embed.add_field(name="<a:uncheck_raveninha:1343471068882669609> *restrict 'Channel id'",
                                value="Specify which channels the bot will be restricted on",
                                inline=False)
                embed.add_field(name="<a:check_raveninha:1343471005078650931> *derestrict 'Channel id'",
                                value="Remove restriction from a certain channel",
                                inline=False)
                embed.add_field(name="📌 Note:", value="Users with **Manage Channels, Administrator** permissions will be"
                                                     " able to Restrict/De-restrict channels", inline=False)

                await message.channel.send(embed=embed)
            else:
                await message.channel.send("You don't have the permissions to use this command.", delete_after=2)

        case _ if command.startswith("restrict"):
            args = command.split()
            if len(args) < 2:
                await message.channel.send("Please provide a channel ID to restrict.")
                return
            channel_id = args[1]
            if message.author == owner or any([
                message.author.guild_permissions.administrator,
                message.author.guild_permissions.manage_channels
            ]):
                await restrict_channel(message, channel_id)
            else:
                await message.channel.send("You don't have the permission to use this command.", delete_after=2)

        case _ if command.startswith("derestrict"):
            args = command.split()
            if len(args) < 2:
                await message.channel.send("Please provide a channel ID to de-restrict.")
                return
            channel_id = args[1]
            if message.author == owner or any([
                message.author.guild_permissions.administrator,
                message.author.guild_permissions.manage_channels
            ]):
                await derestrict_channel(message, channel_id)
            else:
                await message.channel.send("You don't have the permissions to use this command.", delete_after=2)

bot_token = os.getenv("BOT_TOKEN")
client.run(bot_token)
