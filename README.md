Project Overview
This project is a cycle prediction app designed to help users predict their fertility windows and cycle regularity based on their menstrual cycle data. The app uses machine learning to make predictions and provides instant feedback with a strong focus on privacy. The idea behind the app is to give people a privacy-first solution for tracking their menstrual health, while also offering helpful insights into their cycle. The app predicts things like when you're most fertile and whether your cycle is regular, all based on your inputs like cycle number, cycle length, and ovulation day.

Features
Here’s what the app can do:

Fertility Prediction: It predicts your fertility windows based on your cycle data, helping you understand the best times for conception.
Cycle Regularity Prediction: It tells you if your cycle is regular or irregular.
Gemini AI Integration: You can ask Gemini AI questions about your cycle, and it’ll give you personalized answers.
Privacy First: Your data is never shared or sold. It's only used to provide personalized predictions.
Easy-to-Use Interface: The app is simple to use, with color-coded feedback and charts that make it easy to understand your cycle status.
Usage
Once you open the app, you just need to enter your Cycle Number, Cycle Length, and Ovulation Day into the fields. From there, the app will predict your fertility window and let you know whether your cycle is regular or irregular. The results will be shown with easy-to-understand visual feedback like color-coded boxes or charts.

You can also ask Gemini AI any questions you might have about your cycle, and it will provide you with personalized insights based on your inputs.

Deployment
The app is hosted on Streamlit Cloud, so you can access it directly without needing to install anything. Just follow the link to start using the app, and you’re good to go!

Challenges and Lessons Learned
Building this app came with its share of challenges. For one, I had trouble getting Streamlit to work on my computer because my macOS wasn’t up to date. I tried using Colab, but that didn’t work either, so I switched to Streamlit Cloud, which made the deployment process much smoother.

Another challenge was the overfitting issue with the machine learning model. It was initially too specific, so I had to adjust the data and introduce a bit of noise to help the model generalize better.

I also ran into an issue with accidentally exposing the Gemini API key and couldn’t figure out how to use Streamlit secrets properly at first. After some trial and error, I finally got the hang of securely managing the API key with Streamlit’s secrets feature.

Contributing
Feel free to fork this repo and make improvements or add features. Contributions are always welcome! Just let me know at nicole.turpin@duke.edu. Thank you!
