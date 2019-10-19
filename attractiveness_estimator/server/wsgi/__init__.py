from attractiveness_estimator.server.app import create_app


application = create_app()


if __name__ == '__main__':
    application.run()
