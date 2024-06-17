- Use the column telegram_user_id, no need to rename user_id
- usa la siguiente ruta al archivo de datos: ../dataset_diplodatos_completo.csv
- Set the field convesation to Nan for the moment and remove the function identificar_conversaciones
####
- Adjust the filters to match the type of the fields as below:
    - id:   int64
    - commodity:   string
    - source_posted_at:   string
    - trade_type:   string
    - chat_id:    float64
    - telegram_user_id:    float64
    - message_text:   string
####
- Add message lenght to display. Format:
User, date
comodity - trade_type
Message Lengt

#####
- Add a field with the number of line breaks - nessage_lbr
- Add a filter nessage_lbr
- display format:
User, date
Comodity: comodity -  Action: trade_type
Message Lengt: message_length  - LBrs: nessage_lbr

####
- Split the filters in 3 lines and add a title
###