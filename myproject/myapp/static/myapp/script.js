document.getElementById('send-button').addEventListener('click', async () => {
    const userInput = document.getElementById('user-input').value;
    const responseDiv = document.getElementById('response');
    const timeDiv = document.getElementById('time');
    const predictionDiv = document.getElementById('predictions');

    try {
        var num_sentences = 0;
        var curr_text = userInput;
        var new_tokens = []
        var all_probs = []
        while (num_sentences <= 2) {
            const response = await fetch('/api/predict/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ input_text: curr_text })
            });
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }

            const data = await response.json();
            var new_token = data.new_token;
            var token_probs = data.token_probs;
            all_probs.push(token_probs);
            new_tokens.push(new_token);

            console.log(data.new_token);
            console.log(data.token_probs);
            console.log(all_probs);
            curr_text += data.new_token;
            if (new_token == '.' && new_token != "\".") {
                num_sentences++;
            }
            predictionDiv.innerHTML = '';
            // Create a container for the current text and its header
            const currTextContainer = document.createElement('div');
            currTextContainer.classList.add('current-text-container');

            const currtextheader = document.createElement('h2');
            currtextheader.innerText = "Response: ";
            currTextContainer.appendChild(currtextheader);

            const currtextdiv = document.createElement('h6');
            currtextdiv.innerText = curr_text;
            currtextdiv.classList.add('small-header')
            currTextContainer.appendChild(currtextdiv);


            // Append the current text container to the predictions div
            predictionDiv.appendChild(currTextContainer);

            var i = 0
            all_probs.forEach((tokenArray, index) => {

                const tokenDiv = document.createElement('div');
                tokenDiv.classList.add('token-prediction');


                const header = document.createElement('h4');
                header.textContent = `Prediction ${index + 1}`;
                tokenDiv.appendChild(header);

                const currTextHeader = document.createElement('h5'); //header for
                currTextHeader.textContent = 'Current text: '; //text being processed
                currTextHeader.classList.add('small-header');
                tokenDiv.appendChild(currTextHeader);

                const currText = document.createElement('p'); //header for
                currText.textContent = userInput + new_tokens.slice(0, i); //text being processed
                currText.classList.add('small-p');
                tokenDiv.appendChild(currText);

                const probabilitiesheader = document.createElement('h5'); //header for
                probabilitiesheader.textContent = 'Potential next tokens: '; //text being processed
                probabilitiesheader.classList.add('small-header');
                tokenDiv.appendChild(probabilitiesheader);

                // Create the table
                const table = document.createElement('table');
                table.style.width = '100%'; // Optional: Set table width
                table.style.borderCollapse = 'collapse'; // Optional: Collapse table borders

                // Create the table header
                const tableHeader = document.createElement('tr');
                const tokenHeader = document.createElement('th');
                tokenHeader.textContent = 'Token';
                tokenHeader.classList.add('smaller-header')
                const probabilityHeader = document.createElement('th');
                probabilityHeader.textContent = 'Probability';
                probabilityHeader.classList.add('smaller-header')
                tableHeader.appendChild(tokenHeader);
                tableHeader.appendChild(probabilityHeader);
                table.appendChild(tableHeader);

                // Create table rows for each token-probability pair
                all_probs[i].forEach(pair => {
                    const row = document.createElement('tr');

                    const tokenCell = document.createElement('td');
                    tokenCell.textContent = pair[0];
                    tokenCell.classList.add('cell-numbers');
                    tokenCell.style.border = '1px solid black'; // Optional: Add border to cells

                    const probabilityCell = document.createElement('td');
                    probabilityCell.textContent = pair[1];
                    probabilityCell.classList.add('cell-numbers');
                    probabilityCell.style.border = '1px solid black'; // Optional: Add border to cells

                    row.appendChild(tokenCell);
                    row.appendChild(probabilityCell);
                    table.appendChild(row);
                });

                tokenDiv.appendChild(table);
                const selectedtokenheader = document.createElement('h5')// header for selected token
                selectedtokenheader.textContent = 'Selected next token: ';
                selectedtokenheader.classList.add('small-header');
                tokenDiv.appendChild(selectedtokenheader)

                const selectedtoken = document.createElement('p'); //tracks selected token
                selectedtoken.textContent = new_tokens[i]
                selectedtoken.classList.add('small-p');
                tokenDiv.appendChild(selectedtoken);
                predictionDiv.appendChild(tokenDiv);
                i++;
            });

        }
    }
    catch (error) {
        responseDiv.textContent = 'Error: ' + error.message;
    }
});

// Add keydown event listener to textarea
document.getElementById('user-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent newline in textarea
        document.getElementById('send-button').click(); // Trigger button click
    }
});

/*
while llm functions front-end to back-end
front-end: 
keep track of a number of sentences
keep track of the newly generated text (as an array of tokens)
keep track of the all_predictions array 
while that number <= 3
    send the current sentence to the backend 
    parse backend -> new token, its probabilities
    current text.append(new token)
    all_predictions.append(that token's probabilities)
    for(int i = 0; i < current_text.length; i++)
        display current text (initial input + current text array up to last token)
        display probabiltiies (all_predictions[i])
        display selected token (last input of tokens array)
    if that new token is a '.', increment number of sentences
back-end(sentence)
    return selected token, that token's probabilities
*/