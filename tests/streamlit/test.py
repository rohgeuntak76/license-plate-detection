import streamlit as st

# if 'clicked' not in st.session_state:
#     st.session_state.clicked = False

# print(st.session_state)
# def click_button():
#     st.session_state.clicked = True

# # st.button('Click me', on_click=click_button)

# if st.session_state.clicked:
# # if st.button('start'):
#     # The message and nested widget will remain on the page
#     st.write('Button clicked!')
#     st.slider('Select a value')

import streamlit as st

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_state(i):
    st.session_state.stage = i

if st.session_state.stage == 0:
    st.button('Begin', on_click=set_state, args=[1])

if st.session_state.stage >= 1:
    name = st.text_input('Name', on_change=set_state, args=[2])

if st.session_state.stage >= 2:
    st.write(f'Hello {name}!')
    color = st.selectbox(
        'Pick a Color',
        [None, 'red', 'orange', 'green', 'blue', 'violet'],
        on_change=set_state, args=[3]
    )
    if color is None:
        set_state(2)

if st.session_state.stage >= 3:
    st.write(f':{color}[Thank you!]')
    st.button('Start Over', on_click=set_state, args=[0])