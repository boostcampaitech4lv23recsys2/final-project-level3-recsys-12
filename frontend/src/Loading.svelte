<script>
    export let is_loading
    console.log(is_loading)
    const url = "https://cdn-icons-mp4.flaticon.com/512/8722/8722704.mp4"
    
    const width = window.innerWidth
    const height = window.innerHeight

    const icon_width = 100
    const icon_height = 100

    let x = (width - icon_width)/2
    let y = (height - icon_height)/2
</script>

<style>
#loading_backgroud{
    position:fixed;
    top:0;
    left:0;
    width: 100vh;
    height: 100vh;
    background-color:white;
    opacity: 0.5;
    z-index: 100;
}
#loading{
    position:fixed;
    width: 20%;
    height: 20%;
    top: 40%;
    left: 40%;
    z-index:101;
}
</style>

<br>
{#if is_loading==true}
    <div id="loading_backgroud"></div>
    <video style="opacity:1" src={url} id="loading" alt="" autoplay muted loop></video>
{/if}
