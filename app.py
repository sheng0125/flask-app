from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, 這是我的公開網站！"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
