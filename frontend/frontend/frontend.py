import reflex as rx
from typing import List
from pydantic import BaseModel
import httpx
import os

# Читаем адрес бэкенда из env (для Docker), по умолчанию localhost:8000
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Типизация данных ---
class Source(BaseModel):
    doc_id: str
    doc_title: str
    point: str
    text: str
    score: float = 0.0

class ChatMessage(BaseModel):
    role: str
    content: str
    status: str = "answered"
    sources: List[Source] = []

# --- Состояние ---
class State(rx.State):
    messages: List[ChatMessage] = []
    question: str = ""
    is_loading: bool = False
    
    # Состояние для панели документа
    is_panel_open: bool = False
    selected_doc_title: str = ""
    selected_doc_text: str = ""
    highlight_text: str = ""
    is_doc_active: bool = True
    is_doc_loading: bool = False

    def set_question(self, val: str):
        self.question = val

    async def send_message(self):
        q = self.question.strip()
        if not q:
            return

        self.messages.append(ChatMessage(role="user", content=q, sources=[]))
        self.question = ""
        self.is_loading = True
        yield

        history = [{"role": m.role, "content": m.content} for m in self.messages[:-1]]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{API_URL}/ask",
                    json={"question": q, "history": history},
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
            
            sources = [Source(**src) for src in data.get("sources", [])]
            
            self.messages.append(ChatMessage(
                role="assistant",
                content=data.get("answer", "Ошибка формата ответа"),
                status=data.get("status", "answered"),
                sources=sources
            ))
        except Exception as e:
            self.messages.append(ChatMessage(
                role="assistant",
                content=f"Ошибка при соединении с сервером: {str(e)}",
                status="error",
                sources=[]
            ))
        finally:
            self.is_loading = False

    def handle_key(self, key: str):
        if key == "Enter":
            return State.send_message

    # --- Логика открытия документа ---
    async def open_document(self, doc_id: str, highlight: str):
        self.is_panel_open = True
        self.is_doc_loading = True
        self.highlight_text = highlight
        self.selected_doc_text = ""
        self.selected_doc_title = "Загрузка..."
        yield

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{API_URL}/documents/{doc_id}", timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    self.selected_doc_title = data.get("doc_title", "Документ")
                    self.selected_doc_text = data.get("text", "")
                    self.is_doc_active = data.get("is_active", True)
                else:
                    self.selected_doc_title = "Ошибка 404"
                    self.selected_doc_text = "Документ не найден в базе."
        except Exception as e:
            self.selected_doc_title = "Ошибка"
            self.selected_doc_text = str(e)
        finally:
            self.is_doc_loading = False

    def close_panel(self):
        self.is_panel_open = False

# --- UI Компоненты ---

def status_badge(status: str) -> rx.Component:
    return rx.match(
        status,
        ("answered", rx.badge("Полный ответ", background_color="#E6F4EA", color="#1A1A1A", border_radius="12px", padding="4px 8px", font_size="12px", margin_bottom="8px")),
        ("partial", rx.badge("Частичный ответ", background_color="#FEF7E0", color="#1A1A1A", border_radius="12px", padding="4px 8px", font_size="12px", margin_bottom="8px")),
        ("refused_no_data", rx.badge("Нет данных", background_color="#F1F3F4", color="#1A1A1A", border_radius="12px", padding="4px 8px", font_size="12px", margin_bottom="8px")),
        ("refused_offtopic", rx.badge("Не по теме", background_color="#FCE8E6", color="#1A1A1A", border_radius="12px", padding="4px 8px", font_size="12px", margin_bottom="8px")),
        ("refused_illegal", rx.badge("Запрос отклонён", background_color="#FCE8E6", color="#1A1A1A", border_radius="12px", padding="4px 8px", font_size="12px", margin_bottom="8px")),
        ("error", rx.badge("Ошибка", background_color="#FFEBEE", color="#D32F2F", border_radius="12px", padding="4px 8px", font_size="12px", margin_bottom="8px")),
        rx.badge("Неизвестный статус", background_color="#F1F3F4", color="#1A1A1A", border_radius="12px", padding="4px 8px", font_size="12px", margin_bottom="8px")
    )

def sidebar() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text("⚡", font_size="24px", font_weight="bold"),
            rx.text("ПСБ", font_size="24px", font_weight="bold", color="#1A1A1A"),
            padding="20px",
            align_items="center",
        ),
        rx.vstack(
            rx.hstack(rx.icon(tag="message-square", size=18), rx.text("Чат"), padding="10px", border_radius="8px", background_color="#F2F3F5", width="100%", color="#1A1A1A", font_weight="500"),
            rx.hstack(rx.icon(tag="clock", size=18), rx.text("История"), padding="10px", width="100%", color="#666"),
            rx.hstack(rx.icon(tag="star", size=18), rx.text("Избранное"), padding="10px", width="100%", color="#666"),
            rx.hstack(rx.icon(tag="settings", size=18), rx.text("Настройки"), padding="10px", width="100%", color="#666"),
            spacing="2",
            width="100%",
            padding="10px",
        ),
        rx.spacer(),
        rx.hstack(rx.icon(tag="info", size=18), rx.text("Помощь"), padding="10px", width="100%", color="#666"),
        width="250px",
        height="100vh",
        background_color="#FFFFFF",
        border_right="1px solid #E2E8F0",
        padding="10px",
    )

def message_bubble(msg: ChatMessage) -> rx.Component:
    is_user = msg.role == "user"
    is_assistant = msg.role == "assistant"
    
    return rx.box(
        rx.vstack(
            rx.cond(is_assistant, status_badge(msg.status), rx.fragment()),
            rx.text(msg.content, white_space="pre-wrap", color="#1A1A1A"),
            rx.cond(
                is_assistant & (msg.sources.length() > 0),
                rx.vstack(
                    rx.text("Источники:", font_size="12px", color="#666", margin_top="10px"),
                    rx.foreach(
                        msg.sources,
                        lambda src: rx.button(
                            rx.icon(tag="file-text", size=16, color="#666"),
                            rx.text(src.doc_title, font_weight="500", font_size="14px", color="#1A1A1A"),
                            rx.text(f" ({src.point})", font_size="14px", color="#666"),
                            padding="10px 14px",
                            border="1px solid #E2E8F0",
                            border_radius="8px",
                            background_color="#FFFFFF",
                            width="fit-content",
                            cursor="pointer",
                            _hover={"background_color": "#F7FAFC"},
                            on_click=lambda: State.open_document(src.doc_id, src.text)
                        )
                    ),
                    align_items="start",
                    width="100%"
                )
            ),
            align_items="start",
            width="100%",
        ),
        padding="16px 20px",
        background_color=rx.cond(is_user, "#F2F3F5", "#FFFFFF"),
        border_radius="16px",
        border_bottom_right_radius=rx.cond(is_user, "4px", "16px"),
        border_top_left_radius=rx.cond(is_assistant, "4px", "16px"),
        max_width="85%",
        align_self=rx.cond(is_user, "end", "start"),
        margin_bottom="16px",
        border=rx.cond(is_user, "none", "1px solid #E2E8F0"),
    )

def chat_area() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.icon(tag="arrow-left", size=20, color="#666", cursor="pointer"),
            rx.text("Новый чат", font_size="16px", font_weight="500", color="#1A1A1A"),
            width="100%",
            padding="20px",
            border_bottom="1px solid #E2E8F0",
            background_color="#FFFFFF",
        ),
        rx.box(
            rx.cond(
                State.messages.length() == 0,
                rx.vstack(
                    rx.text("Юридический ассистент", font_size="32px", font_weight="bold", margin_bottom="10px", color="#1A1A1A"),
                    rx.text("Надёжный помощник для работы с нормативными документами", color="#666", margin_bottom="30px"),
                    rx.hstack(
                        rx.button("НДС при экспорте", variant="outline", border_radius="20px", color="#1A1A1A", background_color="#FFFFFF", on_click=State.set_question("НДС при экспорте")),
                        rx.button("Условия договора", variant="outline", border_radius="20px", color="#1A1A1A", background_color="#FFFFFF", on_click=State.set_question("Условия договора")),
                        rx.button("Изменения в 2025 году", variant="outline", border_radius="20px", color="#1A1A1A", background_color="#FFFFFF", on_click=State.set_question("Изменения в 2025 году")),
                        spacing="3",
                    ),
                    align_items="start",
                    width="100%",
                    height="100%",
                    justify_content="center",
                ),
                rx.vstack(
                    rx.foreach(State.messages, message_bubble),
                    rx.cond(State.is_loading, rx.spinner(color="gray", size="3", margin="10px")),
                    width="100%",
                    align_items="start",
                )
            ),
            padding="30px",
            width="100%",
            height="calc(100vh - 160px)",
            overflow_y="auto",
            background_color="#FAFAFA",
        ),
        rx.box(
            rx.hstack(
                rx.input(
                    placeholder="Уточните вопрос...",
                    value=State.question,
                    on_change=State.set_question,
                    on_key_down=State.handle_key,
                    width="100%",
                    height="48px",
                    padding="0px 16px",
                    color="#1A1A1A",
                    font_size="16px",
                    border_radius="12px",
                    border="1px solid #E2E8F0",
                    background_color="#FFFFFF",
                    _focus={"border": "1px solid #666", "outline": "none"},
                ),
                rx.button(
                    rx.icon(tag="arrow-right", color="white", size=20),
                    on_click=State.send_message,
                    background_color="#333333",
                    border_radius="12px",
                    width="48px",
                    height="48px",
                    padding="0",
                    cursor="pointer",
                    flex_shrink="0",
                ),
                width="100%",
                align_items="center",
                spacing="3",
            ),
            padding="20px",
            background_color="#FFFFFF",
            border_top="1px solid #E2E8F0",
            width="100%",
        ),
        width="100%",
        height="100vh",
        align_items="start",
    )

# --- Панель просмотра документа (Шторка) ---
def document_panel() -> rx.Component:
    return rx.cond(
        State.is_panel_open,
        rx.box(
            # Затемнение фона (клик по нему закрывает панель)
            rx.box(
                position="fixed", top="0", left="0", width="100vw", height="100vh",
                background_color="rgba(0,0,0,0.4)", z_index="999",
                on_click=State.close_panel,
            ),
            # Сама панель
            rx.vstack(
                rx.hstack(
                    rx.text(State.selected_doc_title, font_weight="bold", font_size="18px", color="#1A1A1A"),
                    rx.cond(~State.is_doc_active, rx.badge("Утратил силу", color_scheme="red", margin_left="10px"), rx.fragment()),
                    rx.spacer(),
                    rx.icon(tag="x", size=24, cursor="pointer", color="#666", on_click=State.close_panel),
                    width="100%", padding="20px", border_bottom="1px solid #E2E8F0", align_items="center"
                ),
                rx.box(
                    rx.cond(
                        State.is_doc_loading,
                        rx.center(rx.spinner(size="3"), width="100%", height="100%"),
                        rx.html(
                            State.selected_doc_text.replace(
                                State.highlight_text,
                                "<mark style='background-color: #FEF7E0; padding: 2px; border-radius: 4px;'>" + State.highlight_text + "</mark>"
                            )
                        )
                    ),
                    padding="30px", overflow_y="auto", width="100%", height="calc(100vh - 70px)",
                    font_size="15px", line_height="1.6", color="#333"
                ),
                position="fixed", top="0", right="0", width="50vw", min_width="400px", height="100vh",
                background_color="white", z_index="1000",
                box_shadow="-4px 0 20px rgba(0,0,0,0.1)",
                align_items="start", spacing="0"
            )
        )
    )

def index() -> rx.Component:
    return rx.hstack(
        sidebar(),
        chat_area(),
        document_panel(),
        width="100%",
        height="100vh",
        spacing="0",
        background_color="#FAFAFA",
    )

app = rx.App()
app.add_page(index)
