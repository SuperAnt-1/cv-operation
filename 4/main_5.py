from gensim.models import KeyedVectors
from gensim.downloader import load

def main():
    # 1. 加载预训练模型
    print("正在加载模型…（首次运行会自动下载）")
    model = load("word2vec-google-news-300")  # 英文 300 维

    # 2. 定义 5 组或以上的词向量算术运算
    tasks = [
        (["king", "woman"], ["man"]),               # king - man + woman
        (["Paris", "Italy"], ["France"]),           # Paris - France + Italy
        (["walking", "swimming"], ["walking"]),     # swimming - walking + walking (同义词测试)
        (["big", "good"], ["small"]),               # big - small + good
        (["happy", "sad"], ["sad"]),                # happy - sad + sad
        (["Tokyo", "Japan"], ["France"]),           # Tokyo - Japan + France (应得 Paris)
    ]

    # 3. 执行并打印结果
    for pos_list, neg_list in tasks:
        result = model.most_similar(positive=pos_list, negative=neg_list, topn=5)
        expr = " + ".join(pos_list)
        if neg_list:
            expr += " - " + " - ".join(neg_list)
        print(f"\n表达式: {expr}")
        for word, score in result:
            print(f"  {word}: {score:.4f}")

if __name__ == "__main__":
    main()
