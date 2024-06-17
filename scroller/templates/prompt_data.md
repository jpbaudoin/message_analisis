Given a pandas dataframe with the following fields:
- message_text: object    
- date_source_posted_at: datetime64

I need transform continous linebraks in message_text to a single line break
Also trim the message
The result should be store in a new field message_text_c1
 

# Token extration:
## lineas simples
- Lines con llave valor. Separacion por dos puntos
    Sacar un listado y procesar
    Standarizar los tokes para las llaves
    Marcar los tokens y tokes para valores
    Tipo de tokens:
        - valores
        - Periodo, accion, variable (e.g profit)
            Entry Price, Mark Price

- Identificacion de monedas
    - Tener lista de monedas
    - Token moneda general
    - Token moneda especifica

