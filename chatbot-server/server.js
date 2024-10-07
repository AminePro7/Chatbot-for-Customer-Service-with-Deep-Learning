const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const axios = require('axios');

const app = express();
const port = 5000;

// Middleware
app.use(bodyParser.json());
app.use(cors());

// Endpoint to handle chat requests
app.post('/api/chat', async (req, res) => {
    const userInput = req.body.message;
    
    try {
        // Make a request to the Python chatbot backend
        const response = await axios.post('http://localhost:5001/chatbot', { message: userInput });
        res.json({ response: response.data });
    } catch (error) {
        console.error('Error communicating with chatbot:', error);
        res.status(500).send('Error communicating with chatbot');
    }
});

app.listen(port, () => {
    console.log(`Node.js server is running on port ${port}`);
});
