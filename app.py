import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai

st.set_page_config(page_title="PKU Food Recommender & Chatbot", layout="centered")
st.title("🍽️ PKU-Friendly Food Recommender + 🤖 Chat with AI")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your PKU food CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV loaded successfully!")

    # --- Normalize data ---
    features = ['Phenylalanine (mg)', 'Protein (g)', 'Energy (kcal)']
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(df[features])

    # --- Cosine Similarity ---
    similarity_matrix = cosine_similarity(norm_data)
    similarity_df = pd.DataFrame(similarity_matrix, index=df['Food'], columns=df['Food'])

    # --- Visualize Similarity Heatmap ---
    st.subheader("🔍 Food Similarity Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_df, cmap='YlGnBu')
    st.pyplot(fig)

    # --- Select a food ---
    st.subheader("🍴 Choose a food to get similar suggestions")
    selected_food = st.selectbox("Select a food:", df['Food'])

    if selected_food:
        selected_index = df[df['Food'] == selected_food].index[0]
        similarity_scores = similarity_matrix[selected_index]
        similar_indices = similarity_scores.argsort()[::-1][1:6]
        recommended = df.iloc[similar_indices]

        st.markdown(f"### Recommended foods similar to **{selected_food}**:")
        st.dataframe(recommended[['Food', 'Phenylalanine (mg)', 'Protein (g)', 'Energy (kcal)']])

        # --- Totals and Comparison ---
        st.subheader("🧮 Nutrient Content of Selected Food")
        row = df[df['Food'] == selected_food].iloc[0]
        phe = row['Phenylalanine (mg)']
        protein = row['Protein (g)']
        energy = row['Energy (kcal)']

        st.metric("Phenylalanine (mg)", f"{phe:.1f}")
        st.metric("Protein (g)", f"{protein:.1f}")
        st.metric("Energy (kcal)", f"{energy:.1f}")

        # --- Progress Bars ---
        max_phe = 250
        max_protein = 20
        max_energy = 800

        st.progress(min(phe / max_phe, 1.0), text="Phe intake")
        st.progress(min(protein / max_protein, 1.0), text="Protein intake")
        st.progress(min(energy / max_energy, 1.0), text="Energy intake")

        # --- Nutrient Bar Chart ---
        st.subheader("📊 Nutrient Comparison Bar Chart")
        chart_data = pd.DataFrame({
            'Nutrient': ['Phenylalanine (mg)', 'Protein (g)', 'Energy (kcal)'],
            'Amount': [phe, protein, energy]
        })
        st.bar_chart(chart_data.set_index('Nutrient'))

# --- Chat with Gemini ---
st.divider()
st.subheader("💬 Chat with PKU AI Assistant")

# ✅ Configure Gemini with updated model name
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as config_error:
    st.error("❌ Failed to configure Gemini API.")
    st.error(f"Configuration error: {config_error}")
    st.stop()

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**AI Assistant:** {message['content']}")

# Clear chat button
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# --- Input field ---
# Use form to handle Enter key properly
with st.form("chat_form", clear_on_submit=True):
    chat_input = st.text_input("Ask a question about PKU nutrition:", key="chat_input_form")
    send_clicked = st.form_submit_button("📤 Send Message", use_container_width=True)

# Handle form submission
if send_clicked and chat_input.strip():
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": chat_input})
    
    with st.spinner("Thinking..."):
        try:
            # Build conversation context
            conversation_context = "You are a helpful assistant specialized in low-phenylalanine diets and PKU nutrition.\n"
            
            # Add CSV data context if available
            if uploaded_file and 'df' in locals():
                conversation_context += "\nAvailable PKU food data from uploaded CSV:\n"
                
                # For questions asking for recommendations or comparisons, include more data
                question_lower = chat_input.lower()
                if any(word in question_lower for word in ['recommend', 'lowest', 'highest', 'compare', 'list', 'show', 'all', 'foods']):
                    # Include full food list with nutritional data
                    conversation_context += "Complete food database:\n"
                    for _, row in df.iterrows():
                        conversation_context += f"{row['Food']}: Phe {row['Phenylalanine (mg)']}mg, Protein {row['Protein (g)']}g, Energy {row['Energy (kcal)']}kcal\n"
                else:
                    # Just show sample for general questions
                    conversation_context += f"Foods in database: {', '.join(df['Food'].head(10).tolist())}"
                    if len(df) > 10:
                        conversation_context += f" and {len(df)-10} more foods"
                
                conversation_context += f"\nTotal foods in database: {len(df)}\n"
                conversation_context += "Each food has data for: Phenylalanine (mg), Protein (g), Energy (kcal)\n"
            
            conversation_context += "\nConversation history:\n"
            
            # Include recent conversation history (last 10 messages to avoid token limits)
            recent_history = st.session_state.chat_history[-10:]
            for msg in recent_history[:-1]:  # Exclude the current message
                if msg["role"] == "user":
                    conversation_context += f"User: {msg['content']}\n"
                else:
                    conversation_context += f"Assistant: {msg['content']}\n"
            
            conversation_context += f"\nCurrent question: {chat_input}"
            
            # Add specific food data if the question mentions foods from the CSV
            if uploaded_file and 'df' in locals():
                # Check if the question mentions any specific foods
                mentioned_foods = []
                chat_lower = chat_input.lower()
                for food in df['Food']:
                    if food.lower() in chat_lower:
                        mentioned_foods.append(food)
                
                # If specific foods are mentioned, add their detailed data
                if mentioned_foods:
                    conversation_context += f"\n\nDetailed nutritional data for mentioned foods:\n"
                    for food in mentioned_foods:
                        food_data = df[df['Food'] == food].iloc[0]
                        conversation_context += f"{food}: Phenylalanine {food_data['Phenylalanine (mg)']}mg, Protein {food_data['Protein (g)']}g, Energy {food_data['Energy (kcal)']}kcal\n"
            
            response = model.generate_content(conversation_context)
            
            # Add AI response to history
            st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            
            # Rerun to show new messages
            st.rerun()
            
        except Exception as e:
            st.error("❌ Gemini API error.")
            st.error(f"Error details: {str(e)}")
            st.info("💡 Try checking if your API key is valid and has access to Gemini models.")