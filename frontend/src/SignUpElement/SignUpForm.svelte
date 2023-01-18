<script>
    const dupcheckbtn = document.querySelector("#dupcheck")
    

    function onDupcheckButtonClick(e){
        const form = document.querySelector("#email_form");
        console.log(form)
    }
    function goToScroll() {
        var location = document.querySelector(".container_title").offsetTop;
        window.scrollTo({top: location, behavior: 'smooth'});
    }
</script>
<style>
#signup_form_wrapper{
    display: flex;
    flex-flow: column nowrap;
    justify-content: center;
    align-items: center;
    height: 100vh;
}
#dupcheck{
    scroll-behavior: smooth;
}
</style>
<div id="signup_form_wrapper">
    <form>
        <input type="email" id="email_form" placeholder="E-mail 주소를 입력하세요.">
        <button id="dupcheck" on:click={onDupcheckButtonClick}>중복 확인</button>
        <button class="h" on:click={goToScroll}>Next!</button>
    </form>
</div>