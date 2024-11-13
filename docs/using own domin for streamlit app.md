You can deploy your app on heroku.com and heroku has an option to use a custom domain in settings. To deploy an app on heroku you have to add two files to your repo on github.

setup.sh

mkdir -p ~/.streamlit/

echo "
[server]\n
headless = true\n
port = $PORT\n
enableCORS = false\n
\n
" > ~/.streamlit/config.toml

Procfile

web: sh setup.sh && streamlit run YourAppNameHere.py

Just change the app name in the Procfile. The rest is a copy paste