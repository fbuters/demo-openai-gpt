# %%
import streamlit as st
import whisper
import openai
import sounddevice as sd
import wavio as wv
import os
from gtts import gTTS
import pygame

# Initialize stuff
initalized = False
if not initalized:
    openai.api_key = os.environ["OPENAI_KEY"]  # Personal key stored in /.zshrc file
    model = whisper.load_model("small.en")
    pygame.mixer.init()
    initalized = True


def record_from_microphone(
    duration: int,
    sample_freq: int = 44100,
):
    """Record from the microphone for period of duration

    Args:
        duration (int): duration of message to record
        sample_freq (int, optional): sample frequency. Defaults to 44100.
    """
    st.write("start recording")
    recording = sd.rec(
        int(duration * sample_freq),
        samplerate=sample_freq,
        channels=1,
    )
    sd.wait()
    st.write("recording is done")
    wv.write("recording0.wav", recording, sample_freq, sampwidth=2)
    pass


def recording_to_text() -> str:
    """Read .wav recordign with name recording0.wav and converts
    speech to text

    Returns:
        str: result from speech to text
    """
    print("start processing")
    audio = whisper.load_audio("recording0.wav")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language="en", fp16=False)
    result = whisper.decode(model, mel, options)
    print("processing done")

    output = result.text  # type: ignore
    print("Speech to text:")
    print(output)

    return output


def call_chatgpt(user_prompt: str, selected_model: str = "text-davinci-003") -> object:
    """_summary_

    Args:
        user_prompt (str): _description_
        selected_model (str, optional): _description_. Defaults to "text-davinci-003".

    Returns:
        object: _description_
    """
    print("Call chatgpt")

    response = openai.Completion.create(
        model=selected_model,
        prompt=user_prompt,
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    print("Response received from chatgpt")
    return response  # type: ignore


def process_response_chatgpt(response: object) -> str:
    """Process and extract the text response from the ChatGPT object

    Args:
        response (object): response object from ChatGPT

    Returns:
        str: answer from ChatGPT
    """
    print("Process response chatgpt")
    output = response["choices"][0]["text"]  # type: ignore
    return output


def text_to_speech(text: str):
    """Convert text to speech

    Args:
        text (str): piece of text to be played out loud.
    """

    tts = gTTS(text)
    tts.save("response.mp3")

    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()

    pass


if __name__ == "__main__":

    st.header("An AI tour-de-force")
    st.markdown(
        """
        This demo app is using [Whisper](https://openai.com/blog/whisper/),
        an open speech-to-text engine.

        It also makes use of [ChatGPT](https://openai.com/blog/chatgpt/) to
        provide us with answers to all our questions.
        
        Finally, the demo uses [Google's text-to-speech engine](https://github.com/pndurette/gTTS) to 
        read out loud the response from ChatGPT
        """
    )
    st.sidebar.subheader("Get started, set the parameters")
    duration = st.sidebar.slider("Recording duration:", 5, 30, 15)
    model_selected = st.sidebar.selectbox(
        "Which OpenAI model GPT model to use?",
        ("text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001"),
    )
    do_text_to_speech = st.sidebar.checkbox("Read answer out loud?")
    record = st.sidebar.button("Start recording!")

    if record:
        st.subheader("Record using the microphone:")
        record_from_microphone(duration)
        result = recording_to_text()
        st.subheader("Results from speech-to-text:")
        st.text(result)
        raw_response_chatgpt = call_chatgpt(result, selected_model=model_selected)  # type: ignore
        processed_response_chatgpt = process_response_chatgpt(raw_response_chatgpt)
        st.subheader("Results from ChatGPT:")
        st.write(processed_response_chatgpt)
        if do_text_to_speech:
            text_to_speech(processed_response_chatgpt)
