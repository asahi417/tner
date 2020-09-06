var entity
var html
var sentence
var i
var start
var end

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
            $('#result').html(jsondata['html'])

        }).fail(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
        });
    })

})