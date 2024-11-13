You need to host the Telegram bot somewhere to keep it running. You have several options:

A. **Run on your CentOS server** (recommended):
```bash
# Create a service file
sudo nano /etc/systemd/system/telegram-bot.service
```

```ini:/etc/systemd/system/telegram-bot.service
[Unit]
Description=NASP Telegram Bot
After=network.target

[Service]
User=your_username
WorkingDirectory=/path/to/your/project
Environment=PATH=/path/to/your/venv/bin
ExecStart=/path/to/your/venv/bin/python src/telegram_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then start it:
```bash
sudo systemctl enable telegram-bot
sudo systemctl start telegram-bot
sudo systemctl status telegram-bot  # Check status
```

B. **Use screen or tmux** (alternative method):
```bash
# Install screen
sudo yum install screen

# Start a new screen session
screen -S telegram-bot

# Run your bot
python src/telegram_bot.py

# Detach from screen: Press Ctrl+A, then D
# Reattach to screen: screen -r telegram-bot
```

The bot needs to run continuously to:
1. Listen for messages
2. Access the vector store
3. Make API calls to Together AI
4. Handle document uploads