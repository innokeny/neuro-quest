from IPython.display import display, clear_output, Markdown, HTML
import ipywidgets as widgets
from datetime import datetime
from pydantic import BaseModel, PrivateAttr
from enum import Enum
import time

from src.engine.engine import Engine
from src.ml.inference.master import MasterResponse

class ActorType(Enum):
    MASTER = "master"
    PLAYER = "player"
    SYSTEM = "system"


class Message(BaseModel):
    actor: ActorType
    text: str = ""
    _timestamp: str = PrivateAttr(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    @property
    def timestamp(self):
        return self._timestamp

class NotebookSession:
    styles: str = """
    <style>
        .master-msg { 
            background-color: #f0f7ff; 
            color: black;
            border-left: 4px solid #4a86e8;
            padding: 10px;
            margin: 10px 0;
            border-radius: 0 10px 10px 0;
        }
        .player-msg {
            background-color: #e6f7e6; 
            color: black;
            border-right: 4px solid #34a853;
            padding: 10px;
            margin: 10px 0;
            text-align: right;
            border-radius: 10px 0 0 10px;
        }
        .system-msg {
            background-color: #f9f9f9;
            color: black;
            font-size: 0.9em;
            padding: 5px;
            margin: 10px 0;
            text-align: center;
            font-style: italic;
        }
        .timestamp {
            font-size: 0.8em;
            color: #999;
            margin-bottom: 5px;
        }
    </style>
    """

    def __init__(
        self,
        engine: Engine,
    ):
        self.engine = engine
        self.history: list[Message] = []
        self.last_master_message: Message | None = None
        self._setup()
    
    def _setup(self):
        self.output = widgets.Output()
        self.input_line = widgets.Text(
            placeholder='Enter your statement',
            layout=widgets.Layout(width='80%')
        )
        self.submit_button = widgets.Button(
            description='Submit',
            button_style='success'
        )

        self.input_line.on_submit(self._submit_callback)

        self.input_container = widgets.HBox(
            [self.input_line, self.submit_button],
            layout=widgets.Layout(margin='10px 0')
        )

        self.window = widgets.VBox([
            self.output,
            self.input_container
        ])

    def _submit_callback(self, _):
        statement: str = self.input_line.value.strip()
        if not statement:
            return
        
        self.input_line.disabled = True
        self.input_line.value = ""
        self.submit_button.disabled = True

        self._add_to_history(Message(
            actor=ActorType.PLAYER,
            text=statement
        ))
        
        try:
            self._render()
            with self.output:
                display(HTML(
                    "<div style='color: gray;'>Master thinking...</div>"
                ))
            response = self.engine.dialog(statement)
            self._add_to_history(Message(
                actor=ActorType.MASTER,
                text=response.text
            ))
        except Exception as e:
            self._add_to_history(Message(
                actor=ActorType.SYSTEM,
                text=f"Error: {e}"
            ))
        
        self.input_line.disabled = False
        self.submit_button.disabled = False
        self._render()


    def _add_to_history(self, message: Message):
        if message.actor == ActorType.MASTER:
            self.last_master_message = message
        self.history.append(message)
    
    def _render(self):
        self._clear()
        with self.output:
            display(HTML(self.styles))
        
        for msg in self.history:
            timestamp = f"<div class='timestamp'>{msg.timestamp}</div>"
            match msg.actor:
                case ActorType.MASTER:
                    display(HTML(
                        f"<div class='master-msg'>"
                        f"{timestamp}"
                        f"<strong>🧙 Master:</strong> {msg.text}"
                        f"</div>"
                    ))
                case ActorType.PLAYER:
                    display(HTML(
                        f"<div class='player-msg'>"
                        f"{timestamp}"
                        f"<strong>👤 Player:</strong> {msg.text}"
                        f"</div>"
                    ))
                case ActorType.SYSTEM:
                    display(HTML(
                        f"<div class='system-msg'>"
                        f"{msg.text}"
                        f"</div>"
                    ))
            
        display(self.input_container)


    def run(self):
        display(self.window)
    
    def _clear(self):
        clear_output(wait=True)