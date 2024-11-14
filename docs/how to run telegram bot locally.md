To integrate **Option A** using `systemd`, let’s go through the setup step-by-step. I’ll also cover how to handle updates to the bot.

### Step-by-Step: Setting Up the Telegram Bot as a `systemd` Service

1. **Create a Service File for the Bot**:
   - Open a new file for the service configuration:
     ```bash
     sudo nano /etc/systemd/system/telegram-bot.service
     ```
   
2. **Define the Service Configuration**:
   - Copy and paste the following configuration, making sure to replace the placeholder paths with the correct ones for your environment:

     ```ini
     [Unit]
     Description=NASP Telegram Bot
     After=network.target

     [Service]
     User=dj  # Replace with your username
     WorkingDirectory=/home/dj/nasp-chatbot  # Adjust to your project path
     Environment=PATH=/home/dj/nasp-chatbot/nasp_env/bin  # Adjust to your virtual environment path
     ExecStart=/home/dj/nasp-chatbot/nasp_env/bin/python src/telegram_bot.py  # Adjust to your main bot file
     Restart=always
     RestartSec=10

     [Install]
     WantedBy=multi-user.target
     ```

3. **Reload the `systemd` Daemon**:
   - After creating or modifying a service file, you must reload `systemd` to recognize the new service:
     ```bash
     sudo systemctl daemon-reload
     ```

4. **Enable and Start the Bot Service**:
   - Enable the bot service so it starts automatically on boot:
     ```bash
     sudo systemctl enable telegram-bot
     ```
   - Start the bot service:
     ```bash
     sudo systemctl start telegram-bot
     ```

5. **Check the Status**:
   - Verify that the bot is running as expected:
     ```bash
     sudo systemctl status telegram-bot
     ```
   - If the bot is running correctly, you’ll see its status as “active (running).”

### Handling Updates to the Bot

When you make updates to the bot’s code, you’ll need to **restart the `systemd` service** to apply the changes. Here’s the process:

1. **Push Your Updates to GitHub** (or apply your changes directly on the server if testing).
2. **Pull Updates on the Server**:
   - SSH into the server and pull the latest changes:
     ```bash
     cd ~/nasp-chatbot
     git pull origin main
     ```
3. **Restart the Bot Service**:
   - Restart the bot service to apply the new code:
     ```bash
     sudo systemctl restart telegram-bot
     ```
   - This command stops the existing bot process and starts it again, loading the latest changes.

### Monitoring and Logs

- To check the status of the bot at any time:
  ```bash
  sudo systemctl status telegram-bot
  ```

- To view logs for debugging:
  ```bash
  sudo journalctl -u telegram-bot -f
  ```

This setup ensures your bot will run continuously and restart automatically if the server reboots or the bot crashes. Let me know if you encounter any issues while setting this up!