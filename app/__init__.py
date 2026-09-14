import os

from flask import Flask

from config import BASE_DIR, SECRET_KEY


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, "templates"),
        static_folder=os.path.join(BASE_DIR, "static"),
    )
    app.config["SECRET_KEY"] = SECRET_KEY

    from app.routes import bp

    app.register_blueprint(bp)
    return app
