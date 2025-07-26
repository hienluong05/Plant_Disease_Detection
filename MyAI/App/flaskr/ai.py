from flask import Blueprint

ai = Blueprint('ai', __name__, url_prefix='/ai')

@ai.route('/ai')
def ai():
    return 'This is plant disease detection AI'

