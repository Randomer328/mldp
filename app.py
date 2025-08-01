import streamlit as st
import numpy as np
import joblib
import pandas as pd


banner_image = "https://github.com/Randomer328/mldp/blob/main/bannerimage.png?raw=true"  
city_sel = "https://github.com/Randomer328/mldp/blob/main/city_sel.png?raw=true"
city_notsel = "https://github.com/Randomer328/mldp/blob/main/city_notsel.png?raw=true"
rural_sel = "https://github.com/Randomer328/mldp/blob/main/rural_sel.png?raw=true"
rural_notsel = "https://github.com/Randomer328/mldp/blob/main/rural_notsel.png?raw=true"
suburb_sel = "https://github.com/Randomer328/mldp/blob/main/Suburb_sel.png?raw=true"
suburb_notsel = "https://github.com/Randomer328/mldp/blob/main/Suburb_notsel.png?raw=true"

if 'locale' not in st.session_state:
    st.session_state['locale'] = None  # Set default value for locale

st.image(banner_image, use_container_width=True)

## banner
banner_text = """
    <div style="position: relative; text-align: center; color: white; font-size: 40px; font-family: 'Poppins', sans-serif;">
        <div style="position: absolute;bottom: 50%; left: 50%; transform: translate(-50%, -50%);">
            🎓 GPA Prediction App
        </div>
    </div>
"""
st.markdown(banner_text, unsafe_allow_html=True)
st.write("Please enter the details below to predict your GPA:")

# Load the trained model
model = joblib.load("lr_model.pkl")


# change colour for score
def get_color(score):
    if score >= 75:
        return '#4CAF50'  # green
    elif score >= 50:
        return '#FFC107'  # amber
    else:
        return '#F44336'  # red
    
@st.cache_data(show_spinner=False) # checks for only changes that effect the graph, so no lag 

# generating the radial chart
def radial_chart(score, label):
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import base64

    # Create a plot for radial chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    angle = (score / 100) * 2 * np.pi  # Calculate the angle for the bar
    radius = 1
    thickness = 0.4

    ax.set_facecolor('#0e1117')  # set background color
    fig.patch.set_facecolor('#0e1117')  # set figure background color

    # Draw the background and the score bar
    ax.barh(radius, 2 * np.pi, left=0, height=thickness, color='lightgray', alpha=0.3)
    ax.barh(radius, angle, left=0, height=thickness, color=get_color(score))

    # Remove axis and ticks for a clean look
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(0, 2)
    ax.set_theta_zero_location("N")  # set zero to the top (North)
    ax.set_theta_direction(-1)  # draw graph clockwise

    # add score and label text in the center of the chart
    ax.text(0, 0, f"{label}\n{score:.1f} / 100", ha='center', va='center',
            fontsize=16, color='white', fontweight='bold')

    plt.axis('off')  # Turn off the axis
    buf = io.BytesIO()  # Use BytesIO to save the image in memory
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)
    buf.seek(0)  # Move the pointer to the start of the buffer
    return base64.b64encode(buf.read()).decode()  # Return the base64-encoded image



col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Math", unsafe_allow_html=True)
    test_score_math = st.number_input("", min_value=0.0, max_value=100.0, step=0.5, key="math")
    math_img = radial_chart(test_score_math, "Math")
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src='data:image/png;base64,{math_img}' width='360'>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col2:
    st.markdown("### Reading", unsafe_allow_html=True)
    test_score_reading = st.number_input("", min_value=0.0, max_value=100.0, step=0.5, key="reading")
    reading_img = radial_chart(test_score_reading, "Reading")
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src='data:image/png;base64,{reading_img}' width='360'>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col3:
    st.markdown("### Science", unsafe_allow_html=True)
    test_score_science = st.number_input("", min_value=0.0, max_value=100.0, step=0.5, key="science")
    science_img = radial_chart(test_score_science, "Science")
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src='data:image/png;base64,{science_img}' width='360'>
        </div>
        """, 
        unsafe_allow_html=True
    )


# locale selection

## CSS for invis buttons
st.markdown(
    """
    <style>
    .element-container:has(style){
        display: none;
    }
    #button-after {
        display: none;
    }
    .element-container:has(#button-after) {
        display: none;
    }
    .element-container:has(#button-after) + div button {
    background-color: transparent;
    color: transparent;
    padding: 12px 32px;
    font-size: 16px;
    border-radius: 10px;
    cursor: pointer;
    position: absolute;
    z-index:4;
    top:45px;
    border: none;
    height: 95px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

part_time_job = st.selectbox("Part-time Job", options=['No', 'Yes'], index=0)

# Create a row of images for locale
st.write("Choose your locale:")
col1, col2, col3 = st.columns(3)
locale = None
city_img = city_notsel
suburb_img = suburb_notsel
rural_img = rural_notsel

# City image
with col1:
    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    if st.button("City",use_container_width=True):
        if city_img == city_notsel:
            city_img = city_sel
            rural_img = rural_notsel
            suburb_img = suburb_notsel
            locale = "City"
            
    st.image(city_img, use_container_width=True)

# Suburban image
with col2:
    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    if st.button("suburban" ,use_container_width=True):
        if suburb_img == suburb_notsel:
            city_img = city_notsel
            rural_img = rural_notsel
            suburb_img = suburb_sel
            locale = "Suburban"
    st.image(suburb_img, use_container_width=True)

# Rural image
with col3:
    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    if st.button("rural",use_container_width=True):
        if rural_img == rural_notsel:
            city_img = city_notsel
            rural_img = rural_sel
            suburb_img = suburb_notsel
            locale = "Rural"
            
    st.image(rural_img, use_container_width=True)

st.write(f"Selected Locale: {locale}")

# input to model
# Combine the inputs into a list
input_data = {
    'TestScore_Math': test_score_math,
    'TestScore_Reading': test_score_reading,
    'TestScore_Science': test_score_science,
    'PartTimeJob': 1 if part_time_job == 'Yes' else 0,
    # 'GoOut': go_out,
    'Locale_City': 1 if st.session_state['locale'] == 'City' else 0,
    'Locale_Rural': 1 if st.session_state['locale'] == 'Rural' else 0,
    'Locale_Suburban': 1 if st.session_state['locale'] == 'Suburban' else 0,
}

input_data = np.array(list(input_data.values()), dtype=float)

# button to trigger prediction

if st.button("Predict GPA"):
    # reset the selected GPA category to None when the button is clicked (optional)
    st.session_state['selected_gpa_category'] = None
       # prepare input data as numpy array
    input_array = np.array(input_data).reshape(1, -1)

    # Make prediction using the model
    prediction = model.predict(input_array)

    # Display the result
    st.markdown(
        """
        <style>
        .gpa-card {
            background-color: #2a2d36;
            padding: 20px;
            border-radius: 10px;
            color: white;
            font-size: 24px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="gpa-card">Predicted GPA: {prediction[0]:.2f}</div>', unsafe_allow_html=True)
        
    # Add spacing
st.markdown(
    """
    <style>
    .spacing {
        margin-bottom: 200px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.markdown('<div class="spacing"></div>', unsafe_allow_html=True)

# personas

# Load the CSV file containing the average data (generated earlier)
gpa_avg = pd.read_csv('gpa_avg.csv')

# Create the tabs for the GPA categories (personas)
gpa_categories = gpa_avg['GPA_category'].unique()

# Add a title above the buttons
st.write("### See average featuers for different gpa categories")

# Check if the GPA category has already been selected in session_state
if 'selected_gpa_category' not in st.session_state:
    st.session_state['selected_gpa_category'] = None  # Default to None if not selected

# Create a row of buttons for selecting the GPA category (persona)
cols = st.columns(len(gpa_categories))  # Create columns dynamically based on number of categories

# Create buttons for each GPA category in a row
for i, category in enumerate(gpa_categories):
    if cols[i].button(f"GPA {category}"):  # Modify the text here
            st.session_state['selected_gpa_category'] = category  # Store the selected category in session_state

# Display the title for the selected persona
if st.session_state['selected_gpa_category']:
    selected_gpa_category = st.session_state['selected_gpa_category']
    st.write(f"### Mean Features for your Persona: {selected_gpa_category}")

    # Filter the data for the selected GPA category
    persona_data = gpa_avg[gpa_avg['GPA_category'] == selected_gpa_category]

    # Separate the columns into different features for the info card
    score_columns = ['TestScore_Math', 'TestScore_Reading', 'TestScore_Science']
    other_columns = ['GPA', 'AttendanceRate', 'StudyHours', 'InternetAccess', 'FreeTime', 'GoOut']

    # Extract the values for the selected persona
    persona_scores = persona_data[score_columns].values.flatten()
    persona_other_features = persona_data[other_columns].values.flatten()

    # Combine the scores and other features into a single card with slight transparency
    info_card = f"""
        <div style="background-color: transparent; padding: 20px; border-radius: 10px; border: 2px solid rgba(255, 255, 255, 0.4); box-shadow: 0 4px 8px rgba(255, 255, 255, 0.4);">
            <h4 style="text-align:center; color:white;">Persona for GPA Category: {selected_gpa_category}</h4>
            <ul style="list-style-type: none; padding: 0; color:white;">
                <li><b>Math:</b> {persona_scores[0]:.2f}</li>
                <li><b>Reading:</b> {persona_scores[1]:.2f}</li>
                <li><b>Science:</b> {persona_scores[2]:.2f}</li>
                <li><b>GPA:</b> {persona_other_features[0]:.2f}</li>
                <li><b>Attendance Rate:</b> {persona_other_features[1]*100:.2f}%</li>
                <li><b>Study Hours/week:</b> {persona_other_features[2]:.2f}</li>
                <li><b>Internet Access:</b> {'Yes' if persona_other_features[3] == 1 else 'No'}</li>
                <li><b>Free Time (1-5):</b> {persona_other_features[4]:.2f}</li>
                <li><b>GoOut (1-5):</b> {persona_other_features[5]:.2f}</li>
            </ul>
        </div>
    """

    # Display the combined info card
    st.markdown(info_card, unsafe_allow_html=True)