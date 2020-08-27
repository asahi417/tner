var entity
var html
var sentence
var i
var return_probability
var model_ckpt

jQuery(document).ready(function () {
    var slider_sentences = $('#max_len')
    slider_sentences.on('change mousemove', function (evt) {
        $('#label_max_len').text('Max sequence length: ' + slider_sentences.val())
    })

    $(document).on('click', '#btn_generate', function (e) {
        if ($('#input_text').val() == "") {
            alert('Insert a text')
            return
        }
        $.ajax({
            url: '/process',
            type: "post",
            contentType: "application/json",
            dataType: "json",
            data: JSON.stringify({
                "input_text": $('#input_text').val(),
                "max_len": $('#max_len').val(),
                "return_probability": $('#return_probability').val()
            }),
            beforeSend: function () {
                $('.overlay').show()
                $('#result').html('')
            },
            complete: function () {
                $('.overlay').hide()
            }
        }).done(function (jsondata, textStatus, jqXHR) {

            entity = jsondata['entity']
            sentence = jsondata['sentence']
            model_ckpt = jsondata['model_ckpt']
            return_probability = jsondata['return_probability']
            html = `<p class="bold">Model ID: ${model_ckpt} </p> <br>`
            html += `<p class="bold">${sentence} </p> <br>`

            i = 0
            for (i = 0; i < entity.length; i++) {
                if (return_probability == true)
                    html += `<p class="bold">${i+1}. ${entity[i]['type']}: ${entity[i]['mention']} (probability: ${entity[i]['probability']}) </p>`
                else
                    html += `<p class="bold">${i+1}. ${entity[i]['type']}: ${entity[i]['mention']} </p>`
            }

            $('#result').html(html)

        }).fail(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
        });
    })

})