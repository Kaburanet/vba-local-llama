import argparse
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 必要に応じてCORSを設定

# モデルのロード（初期化処理）
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
        load_in_8bit=True  # メモリ節約のために8bit量子化
    )
    return tokenizer, model

# グローバル変数
tokenizer = None
model = None

# システムプロンプト（必要に応じて変更）
SYSTEM_PROMPT = """[INST] <<SYS>>
You are a professional colorlist.
<<SYS>>
"""

def generate_response(user_prompt):
    """
    LLMへ入力し、推論結果を返す処理
    """
    # プロンプト構築
    prompt = "<|begin_of_text|>"

    # システムメッセージ
    prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"

    # ユーザーメッセージ
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"

    # アシスタントの開始タグ
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"

    # ストリーマーを使用して逐次出力（今回は最終的なテキストのみ使う想定）
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=1000,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        streamer=streamer
    )

    response_text = ""

    # 別スレッドで生成
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 推論完了まで逐次取得
    for new_text in streamer:
        response_text += new_text

    thread.join()

    return response_text.strip()

@app.route('/query', methods=['POST'])
def query_model():
    """
    ユーザーからPOSTされたJSONを受け取り、
    'query' をLLMに渡して、推論した結果をShift-JISで返す
    """
    data = request.get_json()
    if not data or 'query' not in data:
        # JSONが取れない もしくは 'query' が無い → 400エラー
        return jsonify({'error': 'No query provided'}), 400

    user_query = data['query']
    try:
        # LLMで推論
        llm_response = generate_response(user_query)

        # JSON形式にしてShift-JISエンコードで返す
        response_dict = {"response": llm_response}
        json_str = json.dumps(response_dict, ensure_ascii=False)  # ensure_ascii=Falseで日本語をUTF-16内包のまま

        # Shift-JISにエンコード
        sjis_data = json_str.encode("shift_jis", errors="replace")

        # MIMEタイプ： application/json; charset=shift_jis
        return Response(sjis_data, content_type='application/json; charset=shift_jis')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    global tokenizer, model

    parser = argparse.ArgumentParser(description="LlamaモデルAPIサーバー")
    parser.add_argument("--model_path", type=str, required=True, help="モデルのパスを指定してください")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="サーバーのホスト")
    parser.add_argument("--port", type=int, default=5000, help="サーバーのポート")
    args = parser.parse_args()

    print("モデルをロード中...")
    tokenizer, model = load_model(args.model_path)
    print("モデルのロードが完了しました。")

    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
