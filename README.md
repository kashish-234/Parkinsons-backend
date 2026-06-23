# Backend Project for User Authentication with Firebase

This project is a backend application built using FastAPI that connects to an existing frontend application. It provides user authentication features using Firebase, allowing users to log in via Google or traditional username and password methods.

## Project Structure

```
backend
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── firebase.py
│   ├── middleware
│   │   ├── __init__.py
│   │   └── cors.py
│   ├── models
│   │   ├── __init__.py
│   │   └── user.py
│   ├── routes
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── datasets.py
│   │   ├── models.py
│   │   ├── patients.py
│   │   └── settings.py
│   ├── schemas
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   └── user.py
│   └── services
│       ├── __init__.py
│       ├── auth_service.py
│       └── user_service.py
├── firebase-service-account.json
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Features

- **User Authentication**: Users can authenticate using Google or traditional username/password.
- **Firebase Integration**: Utilizes Firebase for secure authentication and user management.
- **FastAPI Framework**: Built on FastAPI for high performance and easy development.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   - Copy `.env.example` to `.env` and fill in the required values, including Firebase API keys and database URLs.

5. **Run the Application**:
   ```bash
   uvicorn app.main:app --reload
   ```

## Usage

- Access the API at `http://localhost:8000`.
- Use the `/auth/login` endpoint for user login and `/auth/register` for user registration.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.