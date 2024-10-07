import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import { FaPaperPlane, FaUser } from 'react-icons/fa';  // Font Awesome icons

function App() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');

    const sendMessage = async () => {
        if (input.trim() === '') return;

        const userMessage = { text: input, sender: 'user', time: new Date().toLocaleTimeString() };
        setMessages([...messages, userMessage]);

        try {
            const response = await axios.post('http://localhost:5000/api/chat', { message: input });
            const botMessage = { text: response.data.response, sender: 'bot', time: new Date().toLocaleTimeString() };
            setMessages([...messages, userMessage, botMessage]);
        } catch (error) {
            console.error('Error sending message:', error);
        }

        setInput('');
    };

    return (
        <div className="chat-container">
            <div className="chat-box">
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.sender}`}>
                        <div className="message-content">
                            {msg.sender === 'bot' ? <FaUser className="icon bot-icon" /> : <FaUser className="icon user-icon" />}
                            <div className="message-text">
                                {msg.text}
                                <span className="message-time">{msg.time}</span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
            <div className="input-box">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message..."
                />
                <button onClick={sendMessage}>
                    <FaPaperPlane />
                </button>
            </div>
        </div>
    );
}

export default App;
