# 首先，确保安装了必要的库：
# pip install langchain langchain-community langchain-openai beautifulsoup4 chromadb

import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


os.environ["OPENAI_API_KEY"] = "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"

asset_list=["001_bottle","002_bowl","003_plate","004_fluted-block", "005_french-fries",
    "006_hamburg", "007_shoe-box", "008_tray", "009_kettle", "010_pen",
    "011_dustbin", "012_plant-pot", "013_dumbbell-rack", "014_bookcase", "015_laptop",
    "016_oven", "017_calculator", "018_microphone", "019_coaster", "020_hammer",
    "021_cup", "022_cup-with-liquid", "023_tissue-box", "024_scanner", "025_chips-tub",
    "026_pet-collar", "027_table-tennis", "028_roll-paper", "029_olive-oil", "030_drill",
    "031_jam-jar", "032_screwdriver", "033_fork", "034_knife", "035_apple",
    "036_cabinet", "037_box", "038_milk_box", "039_mug", "040_rack",
    "041_shoe", "042_wooden_box", "043_book", "044_microwave", "045_sand-clock",
    "046_alarm-clock", "047_mouse", "048_stapler", "049_shampoo", "050_bell",
    "051_candlestick", "052_dumbbell", "053_teanet", "054_baguette", "055_small-speaker",
    "056_switch", "057_toycar", "058_markpen", "059_pencup", "060_kitchenpot",
    "061_battery", "062_plasticbox", "063_tabletrashbin", "064_msg", "065_soy-sauce",
    "066_vinegar", "067_steamer", "068_boxdrink", "069_vagetable", "070_paymentsign",
    "071_can", "072_electronicscale", "073_rubikscube", "074_displaystand", "075_bread",
    "076_breadbasket", "077_phone", "078_phonestand", "079_remotecontrol", "080_pillbottle",
    "081_playingcards", "082_smallshovel", "083_brush", "084_woodenmallet", "085_gong",
    "086_woodenblock", "087_waterer", "088_wineglass", "089_globe", "090_trophy",
    "091_kettle", "092_notebook", "093_brush-pen", "094_rest", "095_glue",
    "096_cleaner", "097_screen", "098_speaker", "099_fan", "100_seal",
    "101_milk-tea", "102_roller", "103_fruit", "104_board", "105_sauce-can",
    "106_skillet", "107_soap", "108_block", "109_hydrating-oil", "110_basket",
    "111_callbell", "112_tea-box", "113_coffee-box", "114_bottle", "115_perfume",
    "116_keyboard", "117_whiteboard-eraser", "118_tooth-paste", "119_mini-chalkboard", "120_plant"
]


url_list = ["https://robotwin-platform.github.io/doc/objects/{asset}.html"for asset in asset_list]
loaders = [WebBaseLoader(url)for url in url_list]
docs = [loader.load() for loader in loaders]

# 3. 将长文档分割成小块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 4. 将文本块向量化并存入向量数据库
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 5. 创建检索器
retriever = vectorstore.as_retriever()

# 6. 定义提示模板（使用LangChain Hub上的一个流行提示）
prompt = hub.pull("rlm/rag-prompt")

# 7. 定义LLM
llm = ChatOpenAI(model_name="gpt")

# 8. 组装完整的RAG流水线
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 9. 提问！
question = "What is RAG?"
answer = rag_chain.invoke(question)
print(f"问题: {question}")
print(f"答案: {answer}")