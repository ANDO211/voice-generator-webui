###########################################################################################################################################################################
# VG-webui用
import gradio as gr
from vc import vc_interface
from tts import tts_interface
from scripts import download
import os
import json
###########################################################################################################################################################################
# Whisper用
import whisper
import shutil
###########################################################################################################################################################################

# load vits model names
with open('tts/models/model_list.json', 'r', encoding="utf-8") as file:
    global lang_dic
    lang_dic = json.load(file)

vc_models = ['No conversion']
# load rvc model names
os.makedirs('vc/models', exist_ok=True)
vc_model_root = 'vc/models'
vc_models.extend([d for d in os.listdir(vc_model_root) if os.path.isdir(os.path.join(vc_model_root, d))])

def lang_change(lang):
    global model
    global speaker_list

    download.get_vits_model(lang_dic[lang])
    with open(f'tts/models/{lang_dic[lang]}_speakers.txt', "r", encoding="utf-8") as file:
        speaker_list = [line.strip() for line in file.readlines()]

    model = tts_interface.load_model(lang_dic[lang])
    return gr.Dropdown.update(choices=speaker_list)

def vc_change(vcid):
    if vcid != 'No conversion':
        global hubert_model, vc, net_g
        hubert_model = vc_interface.load_hubert()
        vc, net_g = vc_interface.get_vc(vcid)

#ここに音声を
def text2speech(lang, text, sid, vcid, pitch, f0method, length_scale):
    phonemes, tts_audio = tts_interface.generate_speech(model, lang, text, speaker_list.index(sid), False, length_scale)
    if vcid != 'No conversion':
        return phonemes, vc_interface.convert_voice(hubert_model, vc, net_g, tts_audio, vcid, pitch, f0method)
    phonemes2 = swap_chars(phonemes)
    return phonemes2, tts_audio# 発音と音声を出力する関数

def acc2speech(lang, text, sid, vcid, pitch, f0method, length_scale):
    _, tts_audio = tts_interface.generate_speech(model, lang, text, speaker_list.index(sid), True, length_scale)
    if vcid != 'No conversion':
        return vc_interface.convert_voice(hubert_model, vc, net_g, tts_audio, vcid, pitch, f0method)
    return tts_audio

###########################################################################################################################################################################
# 母音変換ルール
def swap_chars(phonemes):
    result = ''
    for char in phonemes:
        if char == 'i':
            result += 'u'
        elif char == 'u':
            result += 'i'
        elif char == 'e':
            result += 'o'
        elif char == 'o':
            result += 'e'
        else:
            result += char
    return result

# アクセントを抽出
def accentgenerate(lang, text, sid, vcid, pitch, f0method, length_scale):
    phonemes, tts_audio = tts_interface.generate_speech(model, lang, text, speaker_list.index(sid), False, length_scale)
    return phonemes

# アクセントから音声を生成
def accent2speech(lang, text, sid, vcid, pitch, f0method, length_scale):
    _, tts_audio = tts_interface.generate_speech(model, lang, text, speaker_list.index(sid), True, length_scale)
    return tts_audio

def boinhenkan(lang, text, sid, vcid, pitch, f0method, length_scale):
    input_audio = gr.Audio(source="microphone", type="filepath", label="録音開始")
    text = transcribe_audio(input_audio)
    phonemes = accentgenerate(lang, text, sid, vcid, pitch, f0method, length_scale)
    henkan_phonemes = swap_chars(phonemes)
    tts_audio1 = accent2speech(lang, henkan_phonemes, sid, vcid, pitch, f0method, length_scale)
    return henkan_phonemes, tts_audio1

###########################################################################################################################################################################
# Whisperモデルのロード
whisper_model = whisper.load_model("base")

def transcribe_audio(filepath):
    try:
        # 保存するディレクトリを指定
        save_dir = "recordings"
        os.makedirs(save_dir, exist_ok=True)

        # 保存するファイルのパスを決定
        save_path = os.path.join(save_dir, "recorded_audio.wav")

        # 音声ファイルを保存
        shutil.copy(filepath, save_path)

        # Whisperで文字起こし
        result = whisper_model.transcribe(save_path, language='ja')

        # 結果のテキストを返す
        return result["text"]
    except Exception as e:
        return str(e)

###########################################################################################################################################################################
# 母音変換のみ最終コード
def boinhenkan2(lang, text, sid, vcid, pitch, f0method, length_scale):
    phonemes = accentgenerate(lang, text, sid, vcid, pitch, f0method, length_scale)
    henkan_phonemes = swap_chars(phonemes)
    tts_audio1 = accent2speech(lang, henkan_phonemes, sid, vcid, pitch, f0method, length_scale)
    return henkan_phonemes, tts_audio1

###########################################################################################################################################################################
# 母音変換+文字起こし最終コード
def boinhenkan3(lang, filepath, sid, vcid, pitch, f0method, length_scale):
    text = transcribe_audio(filepath)
    phonemes = accentgenerate(lang, text, sid, vcid, pitch, f0method, length_scale)
    henkan_phonemes = swap_chars(phonemes)
    tts_audio1 = accent2speech(lang, henkan_phonemes, sid, vcid, pitch, f0method, length_scale)
    return text, henkan_phonemes, tts_audio1

###########################################################################################################################################################################

def save_preset(preset_name, lang_dropdown, sid, vcid, pitch, f0method, speed):
    path = 'ui/speaker_presets.json'
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    data[preset_name] = {}
    data[preset_name]['lang'] = lang_dropdown
    data[preset_name]['sid'] = speaker_list.index(sid)
    data[preset_name]['vcid'] = vcid
    data[preset_name]['pitch'] = pitch
    data[preset_name]['f0method'] = f0method
    data[preset_name]['speed'] = speed

    # 更新されたデータをJSONファイルに書き込み
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def ui():
    with gr.TabItem('Generate'):
        with gr.Row():
            with gr.Column(scale=3):
                input_audio = gr.Audio(source="microphone", type="filepath", label="録音開始")
                boinhenkan3_bt = gr.Button("Generate", variant="primary")
                
                text = gr.Textbox(label="Text", value="こんにちは、世界", lines=8)
                boinhenkan2_bt = gr.Button("Generate From Text", variant="primary")

                phonemes = gr.Textbox(label="Phones", interactive=True, lines=8)
                acc2speech_bt = gr.Button("Generate From Phones", variant="primary")

            with gr.Column():
                lang_dropdown = gr.Dropdown(choices=list(lang_dic.keys()), label="Languages")
                sid = gr.Dropdown(choices=[], label="Speaker")
                lang_dropdown.change(
                    fn=lang_change,
                    inputs=[lang_dropdown],
                    outputs=sid
                )
                speed = gr.Slider(minimum=0.1, maximum=2, step=0.1, label='Speed', value=1)

                vcid = gr.Dropdown(choices=vc_models, label="Voice Conversion", value='No conversion')
                vcid.change(
                    fn=vc_change,
                    inputs=[vcid]
                )
                with gr.Accordion("VC Setteings", open=False):
                    pitch = gr.Slider(minimum=-12, maximum=12, step=1, label='Pitch', value=0)
                    f0method = gr.Radio(label="Pitch Method", choices=["pm", "harvest"], value="pm")

                preset_name = gr.Textbox(label="Preset Name", interactive=True)
                save_preset_bt = gr.Button("Save Preset")
                save_preset_bt.click(
                    fn=save_preset,
                    inputs=[preset_name, lang_dropdown, sid, vcid, pitch, f0method, speed],
                )
        with gr.Row():
            output_audio = gr.Audio(label="Output Audio", type='numpy')
            boinhenkan3_bt.click(
                fn=boinhenkan3,
                inputs=[lang_dropdown, input_audio, sid, vcid, pitch, f0method, speed],
                outputs=[text, phonemes, output_audio]
            )
            boinhenkan2_bt.click(
                fn=boinhenkan2,
                inputs=[lang_dropdown, text, sid, vcid, pitch, f0method, speed],
                outputs=[phonemes, output_audio]
            )
            acc2speech_bt.click(
                fn=acc2speech,
                inputs=[lang_dropdown, phonemes, sid, vcid, pitch, f0method, speed],
                outputs=[output_audio]
            )
