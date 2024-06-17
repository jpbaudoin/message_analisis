# Descripción y objetivo del proyecto:

Este proyecto busca aplicar técnicas de data science para analizar y visualizar la actividad en canales públicos de Telegram que promueven esquemas de pump and dump de criptomonedas. Los alumnos trabajarán con un dataset que incluye mensajes de más de 100 canales, con datos como el ID del chat, la fecha de publicación y el ID del usuario que publicó el mensaje. El objetivo es utilizar métodos de tokenización y análisis de similitudes para identificar posibles colaboraciones entre estos canales, lo que podría indicar que están dirigidos o influenciados por los mismos grupos de personas. Finalmente, se pretende visualizar estos grupos en un grafo utilizando Neo4j, facilitando la identificación de clústeres de canales que trabajan conjuntamente.


# Preguntas clave que el proyecto intenta responder:

- ¿Qué canales de Telegram sobre pump and dumps presentan patrones de mensajes similares y en tiempos cercanos?
- ¿Es posible identificar grupos coordinados detrás de múltiples canales?
- ¿Cómo se distribuyen temporalmente las actividades de pump and dump entre los diferentes canales?
- ¿Cuáles técnicas de procesamiento de lenguaje natural son más efectivas para detectar similitudes en textos de este tipo?
- ¿Cómo se puede visualizar la red de canales para entender mejor sus interconexiones y posibles jerarquías?

# Estructura del Dataset:

El dataset proporcionado contiene las siguientes columnas:
- id: Identificador único del mensaje.
- message_text: Texto del mensaje.
- source_posted_at: Fecha y hora en que se publicó el mensaje.
- trade_type: Tipo de transacción mencionada en el mensaje.
- chat_id: Identificador del chat de Telegram.
- user_id: Identificador del usuario que publicó el mensaje.


# Ideas
- Se puede usar clustering para agrupar mensajes?
- Se puede usar un LLM local para parsear y traducir los mensajes
- Con un LLM local, deberíamos dar algo de jerga de bitcoins?
- Capaz de analisis
    - Por mensajes similares
    - Por usuario en distintos chats
    - Cuanta temporalidad?
    - Nexo con eventos exteriores?


