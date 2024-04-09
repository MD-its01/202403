model_dir = "data/pytorch_model.bin"

from transformers import BertTokenizer, BertForQuestionAnswering, AutoTokenizer, AutoConfig
import torch

config = AutoConfig.from_pretrained("data/config.json")
tokenizer_config = AutoConfig.from_pretrained("data/tokenizer_config.json")

#  東北大の日本語の質問応答モデルを読み込む
model = BertForQuestionAnswering.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", config=config)
model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", config=tokenizer_config)

def predict(quesion, text):
  input_ids = tokenizer.encode(quesion, text)

  #  テキストを1と質問文を00となる配列を生成する。(⽇本語のBERTでは[CLS]が2、[SEP]が3に対応、英語版では[CLS]:101, [SEP]:102に対応）
  token_type_ids = [0 if i <= input_ids.index(3) else 1 for i in range(len(input_ids))]

  #  回答がどの範囲にあるのか開始位置と終了位置のスコアを返す。
  start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

  score = torch.max(start_scores).item() + torch.max(end_scores).item()
  all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

  #  返されたトークンで最も高いテキストを回答として返す
  prediction = ''.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

  prediction = prediction.replace("##", "")
  return prediction, score

print("＝＝＝Bert 読込終了＝＝＝")

text = """人工知能という分野では、コンピュータの黎明期である1950年代から研究開発が行われ続けており、
第1次の「探索と推論」，第2次の「知識表現」というパラダイムで2回のブームが起きたが、
社会が期待する水準に到達しなかったことから各々のブームの後に冬の時代を経験した。
しかし2012年以降、Alexnetの登場で画像処理におけるディープラーニングの有用性が競技会で世界的に認知され、
急速に研究が活発となり、第3次人工知能ブームが到来。
2016年から2017年にかけて、ディープラーニングと強化学習(Q学習、方策勾配法)を導入したAIが完全情報ゲームである囲碁などのトップ棋士、
さらに不完全情報ゲームであるポーカーの世界トップクラスのプレイヤーも破り、
麻雀では「Microsoft Suphx（Super Phoenix）」がオンライン対戦サイト「天鳳」でAIとして初めて十段に到達するなど最先端技術として注目された。
第3次人工知能ブームの主な革命は、自然言語処理、センサーによる画像処理など視覚的側面が特に顕著であるが、社会学、倫理学、技術開発、経済学などの分野にも大きな影響を及ぼしている。
第3次人工知能ブームが続く中、2022年11月30日にOpenAIからリリースされた生成AIであるChatGPTが質問に対する柔軟な回答によって注目を集めたことで、
企業間で生成AIの開発競争が始まるとともに、積極的に実務に応用されるようになった。
この社会現象を第4次人工知能ブームと呼ぶ者も現れている。
一方、スチュアート・ラッセルらの『エージェントアプローチ人工知能』は人工知能の主なリスクとして[致死性自律兵器]、
[監視と説得]、[偏った意思決定]、[雇用への影響]、[セーフティ・クリティカル 安全重視な応用]、[サイバーセキュリティ]を挙げている。
またラッセルらは『ネイチャー』で、人工知能による生物の繁栄と自滅の可能性や倫理的課題についても論じている。
マイクロソフトは「AI for Good Lab」（善きAI研究所）を設置し、eラーニングサービス「DeepLearning.AI」と提携している。
"""

if __name__ == '__main__':
    while True:
        quesion = input()
        if quesion == 'q':
            break
        else:
            prediction, score = predict(quesion, text)
            if score > 5 and prediction != "[CLS]" and prediction != "":
                print("質問「", quesion, "」の回答は:", prediction )
            else:
                print("質問「",quesion,"」の回答は不明です")

print("--終了--")