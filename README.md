# Multi Modal RAG

多模态 RAG 项目

核心功能：
- 解析流水线：布局检测 / 表格结构化 / 公式识别
- 输出文本 + 结构化信息 + 图像切片
- 文本与图像/表格分模态编码，并支持 `alpha` 融合
- 文本库与图像库分开建立索引，保留页码和 bbox
- 支持文本问题跨模态召回图像 / 图表 / 表格
- 重排优先结构相似度，文本语义为辅


## 项目结构

```text
multi_modal_rag_lite/
├── app.py
├── requirements.txt
├── .env.example
└── src/
    ├── config.py
    ├── schemas.py
    ├── utils.py
    ├── parser.py
    ├── embedder.py
    ├── vector_store.py
    ├── retriever.py
    └── generator.py
```

## 安装

```bash
pip install -r requirements.txt
cp .env.example .env
```

## 环境变量

至少配置：

```bash
OPENAI_API_KEY=your_key
```

可选：

```bash
OPENAI_BASE_URL=
OPENAI_CHAT_MODEL=
OPENAI_VISION_MODEL=
OPENAI_EMBED_MODEL=
TABLE_IMAGE_ALPHA=0.65
IMAGE_TEXT_ALPHA=0.55
TOP_K=6
```

## 建索引

```bash
python app.py --mode index --pdf ./demo.pdf
```

## 查询

```bash
python app.py --mode query --question "图表里营收趋势如何？" --top_k 6
```



