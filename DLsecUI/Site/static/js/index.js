$(document).ready(function() {
    $('.sidebar a').on('click', function(e) {
        e.preventDefault();

        // Remove active class from all links
        $('.sidebar a').removeClass('active');
        // Add active class to the clicked link
        $(this).addClass('active');

        // Hide all sections
        $('section').hide();
        // Show the clicked section
        var section = $(this).attr('href');
        $(section).show();
    });
});