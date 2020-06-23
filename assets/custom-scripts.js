$(window).scroll(function(){
$('#dropdown-controls').toggleClass('scrolling', $(window).scrollTop() > $('#dropdown-controls-wrapper').offset().top);
});
