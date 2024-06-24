import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
import sqlite3
import os
import altair as alt

# CSS for styling the app
st.markdown(
    """
    <style>
    .stApp {
        background-color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header section
st.markdown('<div style="display: flex; justify-content: flex-end; margin-top:-70px"><img src="https://i.pinimg.com/originals/4a/73/1f/4a731f6a5480f6ee8b9bfb34168c333b.gif" alt="GIF" width="100%" style="max-width: 400px; margin-right: 160px;"></div>', unsafe_allow_html=True)
st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 29px; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.1);">🌻Інтелектуальна система розпізнавання квітів🌻</p>', unsafe_allow_html=True)
# st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">💐Типи квітів💐</p>', unsafe_allow_html=True)
# st.image("image.png", use_column_width=True)

# Sidebar for image upload method
st.sidebar.title("Метод завантаження зображення")
upload_method = st.sidebar.radio("Будь ласка виберіть модель:", ["Завантажте фото", "Введіть посилання на фото"])

uploaded_image = None  # To store the image uploaded by the user

if upload_method == "Завантажте фото":
    # Upload image from user
    uploaded_image = st.file_uploader("Будь ласка, завантажте зображення квітки:", type=["jpg", "png", "jpeg"])
elif upload_method == "Введіть посилання на фото":
    # Get internet link from user
    st.write("Будь ласка, введіть інтернет-посилання на зображення квітки:")
    image_url = st.text_input("Посилання на зображення")

# Sidebar for model selection
st.sidebar.title("Вибір моделі")
selected_model = st.sidebar.radio("Будь ласка виберіть модель:", ["CNN_model", "VGG16_model", "ResNet_model", "Xception_model", "NASNetMobile_model"])

# Upload image and guess buttons
if uploaded_image is not None or (upload_method == "Введіть посилання на фото" and image_url):
    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">🌼Ваше зображененя🌼</p>', unsafe_allow_html=True)
    if uploaded_image is not None:
        st.image(uploaded_image, caption='', use_column_width=True)
    elif upload_method == "Введіть посилання на фото" and image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='', use_column_width=True)
        except Exception as e:
            st.error("An error occurred while loading the image. Please enter a valid internet link.")

# Model information button
if st.sidebar.button("Інформація про модель"):
    st.markdown(f'<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">🌷{selected_model}🌷</p>', unsafe_allow_html=True)
    if selected_model == "CNN_model":
        st.write("CNN_model - це базова модель згорткової нейронної мережі (CNN). Вона містить згорткові шари, об'єднані шари та повністю з'єднані шари. Часто використовується для базових задач візуальної класифікації.")
    elif selected_model == "VGG16_model":
        st.write("VGG16_model - це 16-шарова модель глибокої згорткової нейронної мережі. Містить послідовні згорткові та об'єднуючі шари. Використовується для таких задач, як візуальна класифікація та розпізнавання об'єктів.")
    elif selected_model == "ResNet_model":
        st.write("ResNet_model - це модель глибокої згорткової нейронної мережі, яка використовує залишкові блоки для полегшення навчання глибоких мереж. Використовується для покращення навчання глибинних мереж.")
    elif selected_model == "Xception_model":
        st.write("Модель Xception: Xception - це модель, яка докорінно змінює архітектуру згорткових нейронних мереж. Вона ефективно витягує ознаки і може бути використана для задач класифікації.")
    elif selected_model == "NASNetMobile_model":
        st.write("Модель NASNetMobile: NASNetMobile - це модель, розроблена шляхом автоматичного пошуку архітектури та оптимізована спеціально для легких і мобільних пристроїв. Вона може бути використана для трансферного навчання для мобільних додатків і портативних пристроїв.")

# Make a guess button
if st.button("Визначити"):
    if upload_method == "Завантажте фото" and uploaded_image is not None:
        image = Image.open(uploaded_image)
    elif upload_method == "Введіть посилання на фото" and image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error("An error occurred while loading the image. Please enter a valid internet link.")

    # Load the model based on the model selected by the user
    model_paths = {
        "CNN_model": 'CNN_model_updated2.h5',
        "VGG16_model": 'VGG16_model_updated2.h5',
        "ResNet_model": 'ResNet50_model_updated2.h5',  # Ensure this is correct
        "Xception_model": 'Xception_model_updated2.h5',
        "NASNetMobile_model": 'NASNetMobile_model_updated2.h5'
    }

    model_path = model_paths.get(selected_model)
    if model_path and os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        st.error(f"Model file {model_path} does not exist.")

    # Prepare the image for the model and make predictions
    if 'image' in locals():
        image = image.resize((224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Make a guess
        prediction = model.predict(image)

        # Show prediction results
        class_names = ["alpine sea holly", "anthurium", "artichoke", "azalea", "balloon flower", "barberton daisy", "bee balm", "bird of paradise", "bishop of llandaff", "black-eyed susan", "blackberry lily", "blanket flower", "bolero deep blue", "bougainvillea", "bromelia", "buttercup", "californian poppy", "camellia", "canna lily", "canterbury bells", "cape flower", "carnation", "cautleya spicata", "clematis", "colt's foot", "columbine", "common dandelion", "common tulip", "corn poppy", "cosmos", "cyclamen", "daffodil", "daisy", "desert-rose", "fire lily", "foxglove", "frangipani", "fritillary", "garden phlox", "gaura", "gazania", "geranium", "giant white arum lily", "globe thistle", "globe-flower", "grape hyacinth", "great masterwort", "hard-leaved pocket orchid", "hibiscus", "hippeastrum", "iris", "japanese anemone", "king protea", "lenten rose", "lilac hibiscus", "lotus", "love in the mist", "magnolia", "mallow", "marigold", "mexican petunia", "monkshood", "moon orchid", "morning glory", "orange dahlia", "osteospermum", "passion flower", "peruvian lily", "petunia", "pincushion flower", "pink primrose", "pink quill", "pink-yellow dahlia", "poinsettia", "primula", "prince of wales feathers", "purple coneflower", "red ginger", "rose", "ruby-lipped cattleya", "siam tulip", "silverbush", "snapdragon", "spear thistle", "spring crocus", "stemless gentian", "sunflower", "sweet pea", "sweet william", "sword lily", "thorn apple", "tiger lily", "toad lily", "tree mallow", "tree poppy", "trumpet creeper", "wallflower", "water lily", "watercress", "wild geranium", "wild pansy", "wild rose", "windflower", "yellow iris"]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.markdown(f'<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">🌷Модельна оцінка🌷</p>', unsafe_allow_html=True)
        st.write(f"Результат прогнозування: {predicted_class}")
        st.write(f"Точність прогнозування: {confidence:.2f}")

        # st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">📊 Ймовірності прогнозування 📊</p>', unsafe_allow_html=True)
        # prediction_df = pd.DataFrame({'Типи квітів': class_names, 'Possibilities': prediction[0]})
        # st.bar_chart(prediction_df.set_index('Типи квітів')),

        # Поріг для відображення ймовірностей
        threshold = 0.05

#############################################

        st.markdown(
    '<p style="background-color: #8a4baf; color: white; font-size: 10px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">📊 Ймовірності прогнозування 📊</p>',
        unsafe_allow_html=True
        )

# Створення DataFrame з прогнозами
        prediction_df = pd.DataFrame({'Типи квітів': class_names, 'Ймовірність': prediction[0]})

# Фільтрація DataFrame за порогом
        filtered_df = prediction_df[prediction_df['Ймовірність'] > threshold]

# Відображення графіку
        st.bar_chart(filtered_df.set_index('Типи квітів'))
######################################


# Connect to the SQLite database
conn = sqlite3.connect('database.db')
c = conn.cursor()

# Create the FlowerTable if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS FlowerTable 
             (id INTEGER PRIMARY KEY AUTOINCREMENT, flower_id INTEGER, flower_name TEXT, flower_date TEXT, add_note TEXT)''')
conn.commit()

# Create the flowers table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS flowers 
             (id INTEGER PRIMARY KEY AUTOINCREMENT, Name TEXT, Location TEXT, Note TEXT, status TEXT)''')
conn.commit()

def issue_flower_form():
    st.header("Призначити квітку")

    flower_options = [row[0] for row in c.execute("SELECT Name FROM flowers WHERE status = 'Available'")]
    flower_Name = st.selectbox("Назва квітки:", flower_options)

    users_options = [row[0] for row in c.execute("SELECT name FROM users")]
    users_name = st.selectbox("Ім'я користувача:", users_options)

    flower_date = st.date_input("Дата випуску:")
    add_note = st.date_input("Термін виконання:")
    submit_button = st.button("Призначити квітку")

    if submit_button:
        issue_flower(flower_Name, users_name, flower_date, add_note)
        st.success(f"{flower_Name} has been issued to {users_name} until {add_note}.")

def issue_flower(flower_Name, users_name, flower_date, add_note):
    c.execute("SELECT id FROM flowers WHERE Name = ?", (flower_Name,))
    flower_id = c.fetchone()[0]
    c.execute("UPDATE flowers SET status = 'issued' WHERE Name = ?", (flower_Name,))
    c.execute("INSERT INTO FlowerTable (flower_id, flower_name, flower_date, add_note) VALUES (?, ?, ?, ?)",
              (flower_id, users_name, flower_date, add_note))
    conn.commit()

def users_registration():
    st.header("Реєстрація користувачів")
    name = st.text_input("Ім'я:")
    email = st.text_input("Email:")
    address = st.text_input("Адреса:")
    submit_button = st.button("Надіслати")

    if submit_button:
        register_users(name, email, address)
        st.success("User registered successfully.")

def register_users(name, email, address):
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, name TEXT, email TEXT, address TEXT)''')
    c.execute("INSERT INTO users (name, email, address) VALUES (?, ?, ?)", (name, email, address))
    conn.commit()

def view_users():
    c.execute('''SELECT FlowerTable.id, flowers.Name, FlowerTable.flower_name, FlowerTable.flower_date, FlowerTable.add_note 
                 FROM FlowerTable INNER JOIN flowers ON FlowerTable.flower_id = flowers.id''')
    data = c.fetchall()
    if data:
        st.write("Список всіх користувачів:")
        df = pd.DataFrame(data, columns=["ID", "Flower Name", "User Name", "Issue Date", "Due Date"])
        st.table(df)
    else:
        st.warning("No users found.")

class Flower:
    def __init__(self, Name, Location, Note, status):
        self.Name = Name
        self.Location = Location
        self.Note = Note
        self.status = status

def add_flower(Name, Location, Note, status):
    flower = Flower(Name, Location, Note, status)
    c.execute("INSERT INTO flowers (Name, Location, Note, status) VALUES (?, ?, ?, ?)", 
              (flower.Name, flower.Location, flower.Note, flower.status))
    conn.commit()

def search_flowers(query):
    c.execute("SELECT * FROM flowers WHERE Name LIKE ? OR Location LIKE ? OR Note LIKE ?", 
              ('%'+query+'%', '%'+query+'%', '%'+query+'%'))
    flowers = c.fetchall()
    return flowers

def delete_flower(Name):
    c.execute('DELETE FROM flowers WHERE Name = ?', (Name,))
    conn.commit()
    st.success('Flower deleted')

def view_flowers():
    c.execute('SELECT * FROM flowers')
    flowers = c.fetchall()
    if not flowers:
        st.write("No flowers found.")
    else:
        flower_table = [[flower[0], flower[1], flower[2], flower[3], flower[4]] for flower in flowers]
        headers = ["ID", "Name", "Location", "Note", "Status"]
        flower_df = pd.DataFrame(flower_table, columns=headers)
        st.write(flower_df.to_html(escape=False), unsafe_allow_html=True)

def get_all_flowers():
    c.execute("SELECT * FROM flowers")
    flowers = c.fetchall()
    return flowers

def view_users_with_flower():
    c.execute('''SELECT FlowerTable.id, flowers.Name, FlowerTable.flower_name, FlowerTable.flower_date, FlowerTable.add_note 
                 FROM FlowerTable INNER JOIN flowers ON FlowerTable.flower_id = flowers.id''')
    data = c.fetchall()
    if data:
        st.write("Список всіх користувачів:")
        df = pd.DataFrame(data, columns=["ID", "Flower Name", "User Name", "Issue Date", "Due Date"])
        st.table(df)
    else:
        st.warning("No users found.")

def main():
    st.title("Система управління квітами")

    menu = ["Додати квітку", "Переглянути квіти", "Шукати квіти", "Видалити квітку", "Реєстрація користувачів", "Переглянути користувачів", "Призначити квітку"]
    choice = st.sidebar.selectbox("Виберіть опцію", menu)

    if choice == "Додати квітку":
        st.subheader("Додати квітку")
        Name = st.text_input("Введіть назву квітки")
        Location = st.text_input("Введіть місцезнаходження")
        Note = st.text_input("Введіть Примітку")
        status = "Available"
        if st.button("Додати"):
            add_flower(Name, Location, Note, status)
            st.success("Flower added.")
    elif choice == "Переглянути квіти":
        st.subheader("Список квітів")
        flowers = get_all_flowers()
        if not flowers:
            st.write("No flowers found.")
        else:
            flower_table = [[flower[0], flower[1], flower[2], flower[3], flower[4]] for flower in flowers]
            headers = ["ID", "Name", "Location", "Note", "Status"]
            flower_df = pd.DataFrame(flower_table, columns=headers)
            st.write(flower_df.to_html(escape=False), unsafe_allow_html=True)
    elif choice == "Шукати квіти":
        st.subheader("Шукати квіти")
        query = st.text_input("Введіть пошуковий запит")
        search_button = st.button("Шукати")
        if query and search_button:
            flowers = search_flowers(query)
            if not flowers:
                st.write("No flowers found.")
            else:
                flower_table = [[flower[0], flower[1], flower[2], flower[3], flower[4]] for flower in flowers]
                headers = ["ID", "Name", "Location", "Note", "Status"]
                flower_df = pd.DataFrame(flower_table, columns=headers)
                st.write(flower_df.to_html(escape=False), unsafe_allow_html=True)
    elif choice == "Видалити квітку":
        st.header("Видалити квітку")
        Name = st.text_input("Введіть назву квітки, яку потрібно видалити:")
        delete_button = st.button("Видалити")
        st.subheader("Список квітів")
        view_flowers()
        if Name and delete_button:
            delete_flower(Name)
            st.success(f"{Name} has been deleted from the library.")
    elif choice == "Реєстрація користувачів":
        users_registration()
    elif choice == "Переглянути користувачів":
        view_users_with_flower()
    elif choice == "Призначити квітку":
        issue_flower_form()

if __name__ == "__main__":
    main()

