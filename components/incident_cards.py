def render_incident_cards(incidents):
    cards_html = ""
    for incident in incidents[::-1]:
        cards_html += f"""
        <div class="card">
            <img src="https://img.icons8.com/fluency/48/000000/alarm.png" class="alarm-icon"/>
            <div class="card-title">{incident['location']}</div>
            <div class="card-subtitle"><strong>CamID:</strong> {incident['cam_id']}</div>
            <div class="card-subtitle"><strong>Timestamp:</strong> {incident['timestamp']}</div>
            <div class="status">Status: <span style="color:red;">{incident['status']} ⏳</span></div>
            <div class="btn-wrapper">
                <button class="custom-btn">Mark as Attended</button>
            </div>
        </div>"""  # Your HTML
        
    return f"""
           <style>
           .card {{
               background-color: #BDB76B;
               border-radius: 10px;
               padding: 20px;
               box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
               width: 300px;
               font-family: 'Arial', sans-serif;
               position: relative;
               margin-bottom: 15px;
            }}
            .alarm-icon {{
                position: absolute;
                top: 20px;
                right: 20px;
                width: 25px;
            }}
            .card-title {{
                font-size: 18px;
                font-weight: bold;
            }}
            .card-subtitle {{
                font-size: 14px;
                color: #333;
                margin-top: 8px;
            }}
            .status {{
                margin-top: 12px;
                color: red;
                font-weight: bold;
            }}
            .btn-wrapper {{
                margin-top: 20px;
                display: flex;
                justify-content: space-between;
            }}
            .custom-btn {{
                background-color: white;
                color: #003366;
                border: 2px solid #003366;
                border-radius: 20px;
                padding: 8px 16px;
                font-weight: bold;
                cursor: pointer;
            }}
            </style>{cards_html}
            """
