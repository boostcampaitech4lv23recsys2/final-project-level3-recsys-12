<script>

    import { push } from 'svelte-spa-router';
	import { member_email, is_login, click_like_item_id } from '../store'

    export let item_id
    let img_url, like_item
    const heart_fill = "https://cdn-icons-png.flaticon.com/512/2589/2589054.png"
    const heart_not_fill = "https://cdn-icons-png.flaticon.com/512/2589/2589197.png"

    if ($is_login) {
        // 이미 좋아요를 누른 경우 처리
        let url = import.meta.env.VITE_SERVER_URL+"/prefer/"+$member_email+"/"+item_id
        
        fetch(url).then((response) => {
            response.json().then((json) => {
                if (json == true) {
                    img_url = heart_fill
                    like_item = true
                }else {
                    img_url = heart_not_fill
                    like_item = false
                }
			})
        })
    }
    else {
        img_url = heart_not_fill
    }

    async function change_like_status() {
        let url
        if ($is_login) {
            if (like_item) {
                let params = {
                    member_email: $member_email,
                    item_id: JSON.stringify(item_id)
                }
                let options = {
                    method: "delete",
                    headers: {
                        "Content-Type": 'application/json'
                    },
                    body: JSON.stringify(params)
                }
                url = import.meta.env.VITE_SERVER_URL + "/delete-prefer/"+$member_email+"/"+item_id
                fetch(url, options).then((response) => {
                    response.json().then((json) => {
                        like_item = false
                    })
                })
            }
            else {
                url = import.meta.env.VITE_SERVER_URL + "/insert-prefer/"+$member_email+"/"+item_id
                fetch(url).then((response) => {
                    response.json().then((json) => {
                        like_item = true
                    })
                })
            }
            
            if (img_url == heart_fill) {
                img_url = heart_not_fill
            } else {
                img_url = heart_fill
            }
        }
        else {
            let response = confirm("로그인이 필요합니다. 로그인 하시겠습니까?")
            if (response) {
                push('/login-user')
                $click_like_item_id = item_id
            }
        }
    }

</script>

<button on:click={() => change_like_status()}
    class="like-wrapper">
    <img class="like-icon" src={img_url} alt="...">
</button>


<style>

	.like-wrapper {
		z-index: 2;
		position: absolute;
		width: 30px;
		height: 30px;
		background-color: white;
		border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0.5rem;
	}

    .like-icon {
        width: 20px;
        height: 20px;
    }

    .like-icon:hover{
        animation-name: shake;
        animation-duration: 2s;
        animation-iteration-count: infinite;
        animation-delay: 0.5s;
    	transition: transform 0.2s linear;
    	transform-origin: 50% 50%;
        transform: scale(1.0)
    }

    @keyframes shake{
        0%{
          transform: rotate(0deg) scale(1.0);
        }
        10%{
          transform: rotate(45deg) scale(1.2);
        }
        20%{
          transform: rotate(-45deg) scale(1.4);
        }
        30%{
          transform: rotate(30deg) scale(1.6);
        }
        40%{
          transform: rotate(-30deg) scale(28);
        }
        50%{
          transform: rotate(10deg) scale(1.6);
        }
        60%{
          transform: rotate(-10deg) scale(1.4);
        }
        70%{
          transform: rotate(0deg) scale(1.2);
        }
        100%{
          transform: rotate(0deg) scale(1.0);
        }
    }
</style>