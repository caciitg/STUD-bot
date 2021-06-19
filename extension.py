from flask import Flask, jsonify, request
import time
from playsound import playsound
# from pydub import AudioSegment
# from pydub.playback import play

app = Flask(__name__)

urls = {}
blacklisted = frozenset(["www.youtube.com", "www.facebook.com", "www.instagram.com"]) 

def url_strip(url):
    if "http://" in url or "https://" in url:
        url = url.replace("https://", '').replace("http://", '').replace('\"', '')
    if "/" in url:
        url = url.split('/', 1)[0]
    return url

@app.route('/send_url', methods=['POST'])
def send_url():
    resp_json = request.get_data()
    params = resp_json.decode()
    url = params.replace("url=", "")
    url = url_strip(url)
    if url in blacklisted:
        playsound('alarm.wav')
        print(url)
        # song = AudioSegment.from_mp3("audio.mp3")
        # play(song)
    urls[url] = urls.get(url, 0) + 1
    print(urls)
    return jsonify({'message': 'success!'}), 200

app.run(host='0.0.0.0', port=5000)




