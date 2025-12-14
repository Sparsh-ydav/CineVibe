# FrontEnd 

Go into the main directory and run `npm i` in the terminal

In the project directory, you can run:

### `npm start`
The page will reload if you make edits.<br />
You will also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.<br />

### `npm run build`

Builds the app for production to the `build` folder.<br />
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.<br />
Your app is ready to be deployed!

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can’t go back!**

# BackEnd

Obtain the dataset from IMDB and then run

`python3 imdb.py` or `python imdb.py`
this will make `imdb_sentiment_model.h5` and `tokenizer.pkl`

`pip install -r requirements.txt` Run this in the terminal under the backend directory.

# Run Flask Server

`python3 mainflask.py` or `python mainflask.py`

start the flask server along with the frontend for the reviews to work
