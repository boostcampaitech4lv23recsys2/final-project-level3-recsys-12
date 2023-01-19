<script>
    import {setContext} from "svelte"

    const dupcheckbtn = document.querySelector("#dupcheck")

    let email
    let is_success

    async function isEmailDup(e){
        if (email == null){
            alert("써라.")
        }else{
            console.log(email)
            let url = "http://localhost:8000/signup/"+email
            await fetch(url).then((response) => {
		    	response.json().then((json) => {
		    		if (response.status == 400){
                        is_success = false
                        alert("중복된 이메일입니다.")
                    }else{
                        is_success = true
                        alert("사용할 수 있는 이메일입니다.")
                    }
		    	})
		    })

        }
    }
    
    function goToScroll(e) {
        e.preventDefault()
        var location = document.querySelector(".container_title").offsetTop;
        window.scrollTo({top: location, behavior: 'smooth'});
    }
    setContext("is_success",is_success)
    setContext("email",email)
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
        <input type="email" id="email_form" placeholder="E-mail 주소를 입력하세요." bind:value="{email}">
        <button id="dupcheck" on:click={isEmailDup}>중복 확인</button>
        <button class="h" on:click={goToScroll}>Next!</button>
    </form>
</div>