import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow_hub as tf_hub
import tensorflow as tf
import tensorflow_text as tf_text
import kagglehub
import spacy
import subprocess



@st.cache_resource
def carregar_modelo():
    path = kagglehub.model_download("inciofilho/testesentimentos1/tensorFlow2/testesentimentos2")
    custom_objects = {'KerasLayer': tf_hub.KerasLayer}
    modelo = load_model(f"{path}/meu_modelo.h5", custom_objects=custom_objects)
    print("Carregado")
    return modelo

@st.cache_resource
def carregar_spacy():
    # Executa o comando como se estivesse no terminal
    subprocess.run(["python3", "-m", "spacy", "download", "pt_core_news_md"])
    # Carregar o modelo de linguagem do spaCy
    nlp = spacy.load("pt_core_news_md")
    return nlp

st.title("Análise de sentimento")

def calcular_sentimento(text):
    print("Iniciou carregamento")
    pred = carregar_modelo().predict([text])
    print("Terminou")
    return "positivo" if pred[0]>=0.5 else "negativo"

def retorna_frase_similar(sentimento: str, frase: str):

  dicionario_sentimento = {
      'positivo': [
          "Que ótimo perceber sentimentos positivos aqui! Continue assim e aproveite o momento.",
          "Fico feliz em ver que o conteúdo reflete emoções positivas. Isso é maravilhoso!",
          "Que bom saber que está se sentindo bem! Vamos celebrar esses sentimentos positivos.",
          "Detectei um sentimento positivo. É ótimo ver essa energia positiva! Continue compartilhando essa vibe.",
          "Sentimentos positivos são sempre bem-vindos! Espero que continue sentindo-se assim.",
          "Que ótimo notar essa positividade. O que acha de refletir sobre o que está gerando esses bons sentimentos?",
          "Que bom ver esse sentimento positivo! Mantenha-se motivado e continue espalhando essa positividade.",
          "Notei um tom positivo aqui. Isso é fantástico! Reconhecer e valorizar esses momentos é muito importante.",
          "Que ótimo ver emoções positivas! Apreciar esses momentos faz toda a diferença.",
          "Que bom ver sentimentos positivos! Vale a pena refletir sobre o que contribuiu para essa sensação e continuar cultivando esses elementos."
      ],
      'negativo': [
          "Há uma indicação de sentimentos negativos aqui. Refletir sobre o que está causando isso pode ser um primeiro passo para encontrar uma solução.",
          "O conteúdo analisado sugere um sentimento negativo.",
          "Identifiquei um tom negativo. Lembre-se de que, mesmo nos momentos difíceis, há maneiras de encontrar apoio e melhorar a situação.",
          "Este conteúdo parece refletir um momento difícil. Considere procurar recursos de apoio, como amigos, família ou profissionais.",
          "Percebo que este conteúdo expressa emoções negativas. É importante reconhecer esses sentimentos e buscar apoio se necessário.",
          "Detectei sentimentos negativos. Talvez seja útil procurar alguém de confiança para conversar sobre isso.",
          "Parece que há algo preocupante aqui.",
          "Os sentimentos negativos detectados são válidos. Não hesite em procurar formas de expressar e processar essas emoções.",
          "Parece que há sentimentos negativos envolvidos. É comum ter dias assim, e buscar aconselhamento pode ser útil.",
          "Notei que o sentimento expressado é negativo. É normal sentir-se assim às vezes. Respirar fundo e dar um tempo para si mesmo pode ajudar.",
          "Parece haver um tom negativo aqui. Lembre-se de que momentos difíceis podem passar e, com o tempo, as coisas podem melhorar."
      ]
  }
  nlp = carregar_spacy()

  lista_para_analise = dicionario_sentimento['positivo'] if sentimento == 'positivo' else dicionario_sentimento['negativo']
  doc_frase = nlp(frase)

  melhor_frase = None
  maior_similaridade = -1

  for frase_candidata in lista_para_analise:
      doc_candidata = nlp(frase_candidata)
      similaridade = doc_frase.similarity(doc_candidata)

      if similaridade > maior_similaridade:
          melhor_frase = frase_candidata
          maior_similaridade = similaridade

  return melhor_frase

texto_input = st.text_input("Digite o texto aqui:")
if st.button("Análise de sentimento"):
    sentimento = calcular_sentimento(texto_input)
    st.write("Análise do sentimento:")
    st.write(sentimento)
    st.write("Frase sugerida:")
    st.write(retorna_frase_similar(sentimento, texto_input))