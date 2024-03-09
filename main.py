from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine, text, Column, String, Integer
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import exists
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score
from werkzeug.security import generate_password_hash, check_password_hash
from contextlib import contextmanager
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, verify_jwt_in_request
import pandas as pd
import pickle
import os

app = Flask(__name__)
CORS(app)

db_user = os.environ.get("DB_USER")
db_password = os.environ.get("DB_PASSWORD")
db_host = os.environ.get("DB_HOST")
db_name = os.environ.get("DB_NAME")
db_path = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}"
engine = create_engine(db_path, echo=False)
Session = sessionmaker(bind=engine)
session = Session()

def load_data_from_db():
    query = "SELECT * FROM clients"
    data = pd.read_sql(query, engine)
    
    return data

def preprocess_data(data):
    x = data.drop('churn', axis=1)
    y = data['churn']
    
    one_hot = make_column_transformer((OneHotEncoder(drop='if_binary'), ['gender', 'geography', 'has_credit_card', 'active_member']), remainder='passthrough', sparse_threshold=0)
    columns = x.columns
    x = one_hot.fit_transform(x)
    one_hot.get_feature_names_out(columns)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=5)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, random_state=5)

    return x_train, x_val, x_test, y_train, y_val, y_test, one_hot

def train_model(x_train, y_train):
    tree_model = DecisionTreeClassifier(max_depth=8, random_state=5)
    tree_model.fit(x_train, y_train)
    
    return tree_model

def export_models(one_hot_encoder, tree_model):
    with open('onehot_model.pkl', 'wb') as file:
        pickle.dump(one_hot_encoder, file)

    with open('tree_model.pkl', 'wb') as file:
        pickle.dump(tree_model, file)

def load_models():
    with open('onehot_model.pkl', 'rb') as file:
        one_hot_encoder = pickle.load(file)

    with open('tree_model.pkl', 'rb') as file:
        tree_model = pickle.load(file)

    return one_hot_encoder, tree_model

def models_exist():
    return os.path.exists('onehot_model.pkl') and os.path.exists('tree_model.pkl')

def calculate_metrics(tree_model, x_test, y_test, x_val, y_val):
    accuracy = round(tree_model.score(x_test, y_test) * 100, 1)
    precision = round(precision_score(y_val, tree_model.predict(x_val)) * 100, 1)
    recall = round(recall_score(y_val, tree_model.predict(x_val)) * 100, 1)

    return accuracy, precision, recall

def predict_and_insert_data_to_sql(session, data_frame, one_hot_encoder, tree_model, token_present):
    transformed_data = one_hot_encoder.transform(data_frame)
    prediction = tree_model.predict(transformed_data)

    original_data = data_frame.copy()

    original_data['churn'] = prediction

    original_data['churn'] = original_data['churn'].astype(int)

    if token_present:
        id_user = get_jwt_identity()

        session.execute(
            text("""
                INSERT INTO new_clients (credit_score, geography, gender, age, ternure, balance,
                                        num_of_products, has_credit_card, active_member, estimated_salary, churn, id_user)
                VALUES (:credit_score, :geography, :gender, :age, :ternure, :balance,
                        :num_of_products, :has_credit_card, :active_member, :estimated_salary, :churn, :id_user)
            """),
            {**original_data.iloc[0].to_dict(), 'id_user': id_user}
        )

        session.commit()
    else:
        pass

    return prediction

def generate_metrics_and_prediction(tree_model, x_test, y_test, x_val, y_val, prediction):
    accuracy, precision, recall = calculate_metrics(tree_model, x_test, y_test, x_val, y_val)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

    if prediction[0] == 0:
        metrics.update({"prediction": "Não cancela o serviço"})
    else:
        metrics.update({"prediction": "Cancela o serviço"})

    return metrics

@app.route("/get_user_clients", methods=["GET"])
@jwt_required()
def get_user_clients():
    try:
        id_user = get_jwt_identity()

        query = text("""
            SELECT * FROM new_clients 
            WHERE id_user = :id_user
            ORDER BY id DESC
            LIMIT 10
        """)
        result = session.execute(query, {"id_user": id_user}).fetchall()

        session.commit()

        df = pd.DataFrame(result)

        user_clients_json = df.to_json(orient="records")

        return user_clients_json, 200

    except Exception as e:
        error_message = f"Internal server error: {str(e)}"
        return jsonify({"error": error_message}), 500

def predict_middleware():
    token_present = "Authorization" in request.headers

    if token_present:
        verify_jwt_in_request()

    return token_present

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        data_frame = pd.DataFrame(data)

        if models_exist():
            one_hot_encoder, tree_model = load_models()
            x_train, x_val, x_test, y_train, y_val, y_test, _ = preprocess_data(load_data_from_db())
        else:
            x_train, x_val, x_test, y_train, y_val, y_test, one_hot_encoder = preprocess_data(load_data_from_db())
            tree_model = train_model(x_train, y_train)
            export_models(one_hot_encoder, tree_model)

        token_present = predict_middleware()

        prediction = predict_and_insert_data_to_sql(session, data_frame, one_hot_encoder, tree_model, token_present)

        metrics = generate_metrics_and_prediction(tree_model, x_test, y_test, x_val, y_val, prediction)

        return jsonify(metrics), 200
    except Exception as e:
        error_message = f"Internal server error: {str(e)}"
        return jsonify({"error": error_message}), 500
    
Base = declarative_base()
    
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)

@contextmanager
def get_session():
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
jwt = JWTManager(app)

def register_user(username, password):
    with get_session() as session:
        user_exists = session.query(exists().where(User.username == username)).scalar()
        if user_exists:
            raise ValueError("Este usuário já existe")

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        session.add(new_user)

        session.commit()

        user_id = new_user.id

        access_token = create_access_token(identity=user_id)
        return access_token

def login_user(username, password):
    with get_session() as session:
        user = session.query(User).filter_by(username=username).first()
        if not user:
            raise ValueError("Usuário não encontrado")

        if not check_password_hash(user.password, password):
            raise ValueError("Senha incorreta")

        access_token = create_access_token(identity=user.id)
        return access_token

@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        access_token = register_user(username, password)
        return jsonify({"access_token": access_token, "message": "Usuário registrado com sucesso"}), 201
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        access_token = login_user(username, password)
        return jsonify({"access_token": access_token, "message": "Login bem-sucedido"}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500
